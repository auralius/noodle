#include <Arduino.h>

#include "noodle.h"
#include "w03.h"
#include "b03.h"
#include "w04.h"
#include "b04.h"
#include "w05.h"
#include "b05.h"

#include <SPI.h>
#include <TFT_eSPI.h>
#include <XPT2046_Touchscreen.h>

// Touchscreen pins
#define XPT2046_IRQ 36   // T_IRQ
#define XPT2046_MOSI 32  // T_DIN
#define XPT2046_MISO 39  // T_OUT
#define XPT2046_CLK 25   // T_CLK
#define XPT2046_CS 33    // T_CS

TFT_eSPI tft = TFT_eSPI();
SPIClass touchscreenSPI = SPIClass(VSPI);
XPT2046_Touchscreen ts(XPT2046_CS, XPT2046_IRQ);

// Touchscreen calibration
constexpr float alpha_x = -0.000f;
constexpr float beta_x = 0.091f;
constexpr float delta_x = -34.949f;
constexpr float alpha_y = 0.066f;
constexpr float beta_y = 0.000f;
constexpr float delta_y = -17.333f;

char INFO[8];
const int IMG_SIZE = 28 * 28;
float *GRID;
float *BUFFER1;
float *BUFFER2;
float *BUFFER3;

#define SCREEN_WIDTH 240
#define SCREEN_HEIGHT 320
#define PENRADIUS 1

static const int16_t L = 84;
static const int16_t L28 = 84 / 28;

static const int16_t W8 = 240 / 8;
static const int16_t W16 = 240 / 16;
static const int16_t W2 = 240 / 2;

// Writing area
byte ROI[8]; // xmin, ymin, xmax, ymax, width, length, center x, center y

// Grayscale colors in 17 steps: 0 to 16
// 17 entries: 16 shades of grey + white
static const uint16_t GREYS[17] = {
    0x0000, // black
    0x0841, // 1
    0x1082, // 2
    0x18C3, // 3
    0x2104, // 4
    0x2945, // 5
    0x3186, // 6
    0x39C7, // 7
    0x4208, // 8
    0x4A49, // 9
    0x528A, // 10
    0x5ACB, // 11
    0x630C, // 12
    0x6B4D, // 13
    0x738E, // 14
    0x7BCF, // 15
    0xFFFF  // white
};

//-----------------------------------------------------------------------------

void progress_hnd(float p)
{
  if (p < 1)
    tft.fillRect(0, L + 1, 240, 3, TFT_BLACK);
  else
    tft.fillRect(0, L + 1, p * 240.0, 3, TFT_BLUE);
}

// Clear the grid data, set all to 0
void reset_grid()
{
  for (int i = 0; i < 28 * 28; ++i)
    GRID[i] = 0.0f;

  // Reset ROI
  for (int16_t i = 0; i < 8; i++)
    ROI[i] = 0;

  ROI[0] = 250; // xmin
  ROI[1] = 250; // ymin
}

// Normalize the numbers in the grids such that they range from 0 to 16
void normalize_grid()
{
  // find the maximum
  float maxval = 0.0f;
  int16_t k = 0;
  for (int16_t i = 0; i < 28; i++)
    for (int16_t j = 0; j < 28; j++)
    {
      if (GRID[k] > maxval)
        maxval = GRID[k];
      k++;
    }

  // normalize such that the maximum is 255.0
  maxval = 255.0f / maxval;
  k = 0;
  for (int16_t i = 0; i < 28; i++)
    for (int16_t j = 0; j < 28; j++)
    {
      GRID[k] = ceil(GRID[k] * maxval); // round instead of floor!
      if (GRID[k] > 255.0)
        GRID[k] = 255.0;
      k++;
    }
}

// Draw the grids in the screen
// The fill-color or the gray-level of each grid is set based on its value
void area_setup()
{
  tft.fillRect(0, 0, 240, 320 - W8, TFT_BLACK);

  for (int16_t i = 0; i < 28; i++)
    for (int16_t j = 0; j < 28; j++)
      tft.fillRect(i * L28, j * L28, L28, L28, GREYS[(uint16_t)GRID[j * 28 + i] / 16]);

  tft.drawRect(0, 0, L, L, TFT_GREEN);
}

// Track ROI, the specific area where the drawing occurs
void track_roi(int16_t xpos, int16_t ypos)
{
  if (xpos - PENRADIUS < ROI[0])
    ROI[0] = xpos - PENRADIUS;
  if (xpos + PENRADIUS > ROI[2])
    ROI[2] = xpos + PENRADIUS;
  if (ypos - PENRADIUS < ROI[1])
    ROI[1] = ypos - PENRADIUS;
  if (ypos + PENRADIUS > ROI[3])
    ROI[3] = ypos + PENRADIUS;

  ROI[4] = ROI[2] - ROI[0];
  ROI[5] = ROI[3] - ROI[1];

  ROI[6] = ROI[0] + ROI[4] / 2;
  ROI[7] = ROI[1] + ROI[5] / 2;
}

// Draw the ROI as a TFT_RED rectangle
void draw_roi()
{
  tft.drawRect(ROI[0], ROI[1], ROI[4], ROI[5], TFT_RED);
  tft.drawCircle(ROI[6], ROI[7], 2, TFT_RED);
}

// Draw the two button at the bottom of the screen
void draw_buttons(const char *label1, const char *label2)
{
  // Add 2 buttons in the bottom: CLEAR and PREDICT
  tft.fillRect(0, 320 - W8, W2, W2, TFT_BLUE);
  tft.fillRect(W2, 320 - W8, W2, W2, TFT_RED);
  tft.setTextSize(2);
  tft.setCursor(6, 320 - W8 + 6);
  tft.print(label1);
  tft.setCursor(W2 + 8, 320 - W8 + 6);
  tft.print(label2);
}

void alloc_buffers()
{
  GRID = (float *)malloc(IMG_SIZE * sizeof(float));
  BUFFER1 = GRID;
  BUFFER2 = (float *)malloc(IMG_SIZE * sizeof(float));
  BUFFER3 = (float *)malloc(14 * 14 * 6 * sizeof(float));
}

byte summarize(float et, float *buf)
{
  float *y = buf; // reuse global memory

  // Find the largest class
  byte label = 0;
  float max_y = 0;

  tft.print(F("--------------------------------------\n"));
  for (byte i = 0; i < 10; i++)
  {
    tft.print(i);
    tft.print(" |");
    for (byte j = 0; j < (byte)round(y[i] * 10); j++)
      tft.print('-');
    tft.print('\n');

    if (y[i] > max_y)
    {
      max_y = y[i];
      label = i;
    }
  }

  tft.print(F("-------------------------------------\nPrediction: "));
  tft.print(label);
  tft.printf(" (p = %.2f)", max_y);
  tft.printf(", in %.1f secs\n", et);

  return label;
}

void predict()
{
  tft.setTextSize(1);
  tft.setCursor(0, W16 * 7);
  tft.println(INFO);

  Conv cnn1;
  cnn1.K = 5;
  cnn1.P = 2;
  cnn1.S = 1; // same padding
  cnn1.weight_fn = "/w01.txt";
  cnn1.bias_fn = "/b01.txt";

  Conv cnn2;
  cnn2.K = 5;
  cnn2.P = 0;
  cnn2.S = 1; // valid padding
  cnn2.weight_fn = "/w02.txt";
  cnn2.bias_fn = "/b02.txt";

  Pool pool;
  pool.M = 2;
  pool.T = 2;

  FCNMem fcn_mem1; // weights + biases are in memory
  fcn_mem1.weight = w03;
  fcn_mem1.bias = b03;
  fcn_mem1.act = ACT_RELU;

  FCNMem fcn_mem2; // weights + biases are in memory
  fcn_mem2.weight = w04;
  fcn_mem2.bias = b04;
  fcn_mem2.act = ACT_RELU;

  FCNMem fcn_mem3; // weights + biases are in memory
  fcn_mem3.weight = w05;
  fcn_mem3.bias = b05;
  fcn_mem3.act = ACT_SOFTMAX;

  unsigned long st = micros(); // timer starts
  uint16_t V;

  tft.println(F("Conv #1 ..."));
  V = noodle_conv_float(BUFFER1, 1, 6, BUFFER3, 28, cnn1, pool, NULL);

  tft.println(F("Conv #2 ..."));
  V = noodle_conv_float(BUFFER3, 6, 16, BUFFER1, V, cnn2, pool, NULL);

  tft.println(F("Flattening ..."));
  V = noodle_flat(BUFFER1, BUFFER3, V, 16);

  tft.println(F("Dense #1 ..."));
  V = noodle_fcn(BUFFER3, V, 120, BUFFER1, fcn_mem1, NULL);

  tft.println(F("Dense #2 ..."));
  V = noodle_fcn(BUFFER1, V, 84, BUFFER3, fcn_mem2, NULL);

  tft.println(F("Dense #3 ..."));
  V = noodle_fcn(BUFFER3, V, 10, BUFFER1, fcn_mem3, NULL);

  float et = (float)(micros() - st) * 1e-6; // timer stops

  byte res = summarize(et, BUFFER1);
}

void setup()
{
  Serial.begin(115200);

  // while (!noodle_sd_init(14, 15, 2)) { // 1-bit pins for ESP32-CAM (AI-Thinker)
  while (!noodle_sd_init())
  { // 4-bit
    delay(500);
    tft.println(".");
  }
  tft.println(F("FFAT OK!"));

  touchscreenSPI.begin(XPT2046_CLK, XPT2046_MISO, XPT2046_MOSI, XPT2046_CS);
  ts.begin(touchscreenSPI);
  tft.init();

  tft.setRotation(2);
  ts.setRotation(2);

  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_GREEN);

  alloc_buffers();

  reset_grid();
  area_setup();
  draw_buttons("CLEAR", "PREDICT");

  noodle_setup_temp_buffers((void *)BUFFER2);
  noodle_read_top_line("/info.txt", INFO, 8);
}

void loop()
{
  int16_t xpos, ypos;
  // Checks if Touchscreen was touched, and prints X, Y and Pressure (Z) info on the TFT display and Serial Monitor
  if (ts.tirqTouched() && ts.touched())
  {
    TS_Point tp = ts.getPoint();

    // Calibrate Touchscreen points with map function to the correct width and height
    xpos = alpha_y * tp.x + beta_y * tp.y + delta_y;
    if (xpos < 0)
      xpos = 0;
    else if (xpos > SCREEN_WIDTH)
      xpos = SCREEN_WIDTH;

    ypos = alpha_x * tp.x + beta_x * tp.y + delta_x;
    if (ypos < 0)
      ypos = 0;
    else if (ypos > SCREEN_HEIGHT)
      ypos = SCREEN_HEIGHT;

    // are we in drawing area ?
    if (((ypos - PENRADIUS) > 0) && ((ypos + PENRADIUS) < L) && ((xpos - PENRADIUS) > 0) && ((xpos + PENRADIUS) < L))
    {
      tft.fillCircle(xpos, ypos, PENRADIUS, TFT_BLUE);
      track_roi(xpos, ypos);
    }

    // CLEAR?
    if ((ypos > tft.height() - W8) && (xpos < W2))
    {
      reset_grid();
      area_setup();
    }

    // PREDICT?
    if ((ypos > tft.height() - W8) && (xpos > W2))
    {
      draw_roi();
      int16_t sum[3] = {0, 0, 0};
      for (byte h = 0; h < 2; h++)
      {
        for (byte i = 0; i < 28; i++)
        {
          for (byte j = 0; j < 28; j++)
          {
            for (byte k = 0; k < L28; k++)
            {
              for (byte l = 0; l < L28; l++)
              {
                int16_t x = i * L28 + k;
                int16_t y = j * L28 + l;

                uint16_t pixel = tft.readPixel(x, y);

                if (pixel == TFT_BLUE)
                {
                  if (h == 0)
                  {
                    sum[0] += x;
                    sum[1] += y;
                    sum[2] += 1;
                  }
                  else
                  {
                    int16_t cx = sum[0] / sum[2];
                    int16_t cy = sum[1] / sum[2];

                    float s = 60.0 / (float)ROI[5];
                    int16_t x_ = s * (float)(x - cx) + (float)(L * 0.5f);
                    int16_t y_ = s * (float)(y - cy) + (float)(L * 0.5f);

                    if ((x_ >= 0) && (x_ < L) && (y_ >= 0) && (y_ < L))
                      GRID[y_ / L28 * 28 + (x_ / L28)] = GRID[y_ / L28 * 28 + (x_ / L28)] + 1;
                  }
                }
              }
            }
          }
        }
        // if  (h == 0)
        //   tft.drawCircle((int16_t)(sum[0] / sum[2]), (int16_t)(sum[1] / sum[2]), 3, TFT_GREEN);
      }

      normalize_grid();
      area_setup();
      predict();
      reset_grid();
    }
  }
}

