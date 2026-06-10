// Last tested on ESP32-CYD, June 10, 2026
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

TFT_eSPI tft = TFT_eSPI();
SPIClass touchscreenSPI = SPIClass(VSPI);
XPT2046_Touchscreen ts(XPT2046_CS, XPT2046_IRQ);

// Touchscreen calibration
constexpr float alpha_x = -0.000f;
constexpr float beta_x  =  0.091f;
constexpr float delta_x = -34.949f;
constexpr float alpha_y =  0.066f;
constexpr float beta_y  =  0.000f;
constexpr float delta_y = -17.333f;

char INFO[8];

#define SCREEN_WIDTH  240
#define SCREEN_HEIGHT 320
#define PENRADIUS 1

static const uint16_t IMG_W = 28;
static const uint16_t IMG_H = 28;
static const uint16_t IMG_SIZE = IMG_W * IMG_H;

static const int16_t L   = 84;
static const int16_t L28 = 84 / 28;

static const int16_t W8  = 240 / 8;
static const int16_t W16 = 240 / 16;
static const int16_t W2  = 240 / 2;

// Writing area
byte ROI[8]; // xmin, ymin, xmax, ymax, width, length, center x, center y

// Smart grow-only buffers.
// A holds the input grid first.
// B is the first output buffer.
// Then we manually ping-pong A <-> B.
static NoodleBuffer A;
static NoodleBuffer B;

// Grayscale colors in 17 steps: 0 to 16
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

static inline void swap_buffers(NoodleBuffer *&in, NoodleBuffer *&out)
{
  NoodleBuffer *tmp = in;
  in = out;
  out = tmp;
}

static inline float *grid_data()
{
  return A.data;
}

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
  float *grid = noodle_buffer_require(&A, IMG_SIZE);
  if (!grid) return;

  for (uint16_t i = 0; i < IMG_SIZE; ++i)
    grid[i] = 0.0f;

  // Reset ROI
  for (int16_t i = 0; i < 8; i++)
    ROI[i] = 0;

  ROI[0] = 250; // xmin
  ROI[1] = 250; // ymin
}

// Normalize the numbers in the grids such that they range from 0 to 255
void normalize_grid()
{
  float *grid = grid_data();
  if (!grid) return;

  float maxval = 0.0f;
  for (uint16_t k = 0; k < IMG_SIZE; k++)
    if (grid[k] > maxval)
      maxval = grid[k];

  if (maxval <= 0.0f)
    return;

  const float scale = 255.0f / maxval;

  for (uint16_t k = 0; k < IMG_SIZE; k++)
  {
    grid[k] = ceil(grid[k] * scale);
    if (grid[k] > 255.0f)
      grid[k] = 255.0f;
  }
}

// Draw the grids in the screen
// The fill-color or the gray-level of each grid is set based on its value
void area_setup()
{
  float *grid = grid_data();
  if (!grid) return;

  tft.fillRect(0, 0, 240, 320 - W8, TFT_BLACK);

  for (int16_t i = 0; i < 28; i++)
    for (int16_t j = 0; j < 28; j++)
    {
      uint16_t idx = (uint16_t)grid[j * 28 + i] / 16;
      if (idx > 16) idx = 16;
      tft.fillRect(i * L28, j * L28, L28, L28, GREYS[idx]);
    }

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
  tft.fillRect(0, 320 - W8, W2, W2, TFT_BLUE);
  tft.fillRect(W2, 320 - W8, W2, W2, TFT_RED);
  tft.setTextSize(2);
  tft.setCursor(6, 320 - W8 + 6);
  tft.print(label1);
  tft.setCursor(W2 + 8, 320 - W8 + 6);
  tft.print(label2);
}

void init_buffers()
{
  noodle_buffer_init(&A);
  noodle_buffer_init(&B);

  if (!noodle_buffer_require(&A, IMG_SIZE)) {
    Serial.println("[ERR] A allocation failed");
    tft.println("[ERR] A allocation failed");
    while (1) delay(1000);
  }
}

byte summarize(float et, NoodleBuffer *buf)
{
  uint16_t label = 0;
  float max_y = 0.0f;

  noodle_find_max(buf, 10, max_y, label);

  tft.print(F("--------------------------------------\n"));
  for (byte i = 0; i < 10; i++)
  {
    const float p = buf->data[i];

    tft.print(i);
    tft.print(" |");
    for (byte j = 0; j < (byte)round(p * 10); j++)
      tft.print('-');
    tft.print('\n');
  }

  tft.print(F("-------------------------------------\nPrediction: "));
  tft.print(label);
  tft.printf(" (p = %.2f)", max_y);
  tft.printf(", in %.1f secs\n", et);

  return (byte)label;
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
  cnn1.weight_fn = "/w01.bin";
  cnn1.bias_fn = "/b01.bin";

  Conv cnn2;
  cnn2.K = 5;
  cnn2.P = 0;
  cnn2.S = 1; // valid padding
  cnn2.weight_fn = "/w02.bin";
  cnn2.bias_fn = "/b02.bin";

  Pool pool;
  pool.M = 2;
  pool.T = 2;

  FCNMem fcn_mem1;
  fcn_mem1.weight = w03;
  fcn_mem1.bias = b03;
  fcn_mem1.act = ACT_RELU;

  FCNMem fcn_mem2;
  fcn_mem2.weight = w04;
  fcn_mem2.bias = b04;
  fcn_mem2.act = ACT_RELU;

  FCNMem fcn_mem3;
  fcn_mem3.weight = w05;
  fcn_mem3.bias = b05;
  fcn_mem3.act = ACT_SOFTMAX;

  unsigned long st = micros();

  NoodleBuffer *in  = &A;
  NoodleBuffer *out = &B;

  tft.println(F("Conv #1 ..."));
  Serial.println(F("Conv #1 ..."));
  uint16_t V = noodle_conv_float(in, 1, 6, out, 28, cnn1, pool, NULL);
  swap_buffers(in, out);

  tft.println(F("Conv #2 ..."));
  V = noodle_conv_float(in, 6, 16, out, V, cnn2, pool, NULL);
  swap_buffers(in, out);

  tft.println(F("Flattening ..."));
  V = noodle_flat(in, out, V, 16);
  swap_buffers(in, out);

  tft.println(F("Dense #1 ..."));
  V = noodle_fcn(in, V, 120, out, fcn_mem1, NULL);
  swap_buffers(in, out);

  tft.println(F("Dense #2 ..."));
  V = noodle_fcn(in, V, 84, out, fcn_mem2, NULL);
  swap_buffers(in, out);

  tft.println(F("Dense #3 ..."));
  V = noodle_fcn(in, V, 10, out, fcn_mem3, NULL);
  swap_buffers(in, out);

  float et = (float)(micros() - st) * 1e-6;

  summarize(et, in);
}

void setup()
{
  Serial.begin(115200);

  tft.init();
  tft.setRotation(2);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_GREEN);

  touchscreenSPI.begin(XPT2046_CLK, XPT2046_MISO, XPT2046_MOSI, XPT2046_CS);
  ts.begin(touchscreenSPI);
  ts.setRotation(2);
  delay(3000);

  while (!noodle_fs_init())
  {
    delay(500);
    Serial.println(".");
    tft.println(".");
  }

  Serial.println(F("FFAT OK!"));

  init_buffers();

  reset_grid();
  area_setup();
  draw_buttons("CLEAR", "PREDICT");
}

void loop()
{
  int16_t xpos, ypos;

  if (ts.tirqTouched() && ts.touched())
  {
    TS_Point tp = ts.getPoint();

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

    if (((ypos - PENRADIUS) > 0) &&
        ((ypos + PENRADIUS) < L) &&
        ((xpos - PENRADIUS) > 0) &&
        ((xpos + PENRADIUS) < L))
    {
      tft.fillCircle(xpos, ypos, PENRADIUS, TFT_BLUE);
      track_roi(xpos, ypos);
    }

    if ((ypos > tft.height() - W8) && (xpos < W2))
    {
      reset_grid();
      area_setup();
    }

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
                    {
                      float *grid = grid_data();
                      if (grid) {
                        const uint16_t idx = (y_ / L28) * 28 + (x_ / L28);
                        grid[idx] = grid[idx] + 1.0f;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      normalize_grid();
      area_setup();
      predict();
      reset_grid();
    }
  }
}
