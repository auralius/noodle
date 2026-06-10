/**
 * Auralius Manurung -- Universitas Telkom, Bandung
 *   auralius.manurung@ieee.org
 * Lisa Kristiana -- ITENAS, Bandung
 *   lisa@itenas.ac.id
 * 
 * Last tested on Uno R3 with 2.4" TFT LCD shield on June 10, 2026.
 */
#include <Arduino.h>
#include <MCUFRIEND_kbv.h>
#include <TouchScreen.h>
#include "noodle.h"

MCUFRIEND_kbv tft;

// Typically there are 2 variant pin configurations for the touchscreen
// const uint16_t XP = 6, XM = A2, YP = A1, YM = 7; // variant 1
const uint16_t XP = 8, YP = A3, XM = A2, YM = 9; // variant 2

// Touch screen boundaries, obtained by calibration
const int TS_LEFT = 124, TS_RT = 912, TS_TOP = 898, TS_BOT = 81;
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);
TSPoint tp;

#define SD_CS 10
#define MINPRESSURE 200
#define MAXPRESSURE 1000
#define PENRADIUS 1

const int16_t L = 96;
const int16_t L16 = 96 / 16;

const int16_t W8 = 240 / 8;
const int16_t W16 = 240 / 16;
const int16_t W2 = 240 / 2;

// Writing area
byte ROI[8]; // xmin, ymin, xmax, ymax, width, length, center x, center y

// The discretized drawing area: 16x16 grids, max value of each grid is 255
byte GRID[16 * 16];

// New Noodle smart buffers for FCN outputs.
// The first FCN reads GRID directly and writes to A.
// The second FCN reads A and writes to B.
static NoodleBuffer A;
static NoodleBuffer B;

#define NORMALIZE_0_1

// Grayscale colors in 17 steps: 0 to 16
const uint16_t GREYS[] PROGMEM = {
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

static inline uint16_t grey(uint8_t i)
{
  return pgm_read_word(&GREYS[i]);
}

void reset_grid()
{
  for (int16_t i = 0; i < 16; i++)
    for (int16_t j = 0; j < 16; j++)
      GRID[i * 16 + j] = 0;

  for (int16_t i = 0; i < 8; i++)
    ROI[i] = 0;

  ROI[0] = 250;
  ROI[1] = 250;
}

void normalize_grid()
{
  int16_t maxval = 0;
  for (int16_t i = 0; i < 16; i++)
    for (int16_t j = 0; j < 16; j++)
      if (GRID[i * 16 + j] > maxval)
        maxval = GRID[i * 16 + j];

  if (maxval == 0)
    return;

  for (int16_t i = 0; i < 16; i++)
    for (int16_t j = 0; j < 16; j++)
      GRID[i * 16 + j] = round((float)GRID[i * 16 + j] / (float)maxval * 255.0);
}

void area_setup()
{
  tft.fillRect(0, 0, 240, 320 - W8, TFT_BLACK);

  for (int16_t i = 0; i < 16; i++)
    for (int16_t j = 0; j < 16; j++)
    {
      uint8_t idx = GRID[j * 16 + i] >> 4;
      if (idx > 16)
        idx = 16;
      tft.fillRect(i * L16, j * L16, L16, L16, grey(idx));
    }

  tft.drawRect(0, 0, L, L, TFT_GREEN);

  tft.setTextSize(1);
  tft.setCursor(W2, 10);
  tft.print(F("2-layer FCN UNO R3\n"));
}

void track_roi(int16_t xpos, int16_t ypos)
{
  if (xpos - PENRADIUS < ROI[0]) ROI[0] = xpos - PENRADIUS;
  if (xpos + PENRADIUS > ROI[2]) ROI[2] = xpos + PENRADIUS;
  if (ypos - PENRADIUS < ROI[1]) ROI[1] = ypos - PENRADIUS;
  if (ypos + PENRADIUS > ROI[3]) ROI[3] = ypos + PENRADIUS;

  ROI[4] = ROI[2] - ROI[0];
  ROI[5] = ROI[3] - ROI[1];

  ROI[6] = ROI[0] + ROI[4] / 2;
  ROI[7] = ROI[1] + ROI[5] / 2;
}

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

void setup(void)
{
  tft.reset();
  tft.begin(tft.readID());

  tft.fillScreen(TFT_BLACK);

  tft.setCursor(0, 10);
  tft.setTextSize(1);
  tft.setTextColor(TFT_GREEN);

  noodle_buffer_init(&A);
  noodle_buffer_init(&B);

  if (!noodle_fs_init(SD_CS))
  {
    tft.println(F("cannot start SD"));
    while (1)
      ;
  }

  reset_grid();
  area_setup();
  draw_buttons("CLEAR", "PREDICT");
}

void progress_hnd(float p)
{
  if (p < 0.01)
    tft.fillRect(0, L + 1, 240, 3, TFT_BLACK);
  else
    tft.fillRect(0, L + 1, p * 240.0, 3, TFT_BLUE);
}

void summarize(float et, NoodleBuffer *output)
{
  tft.setCursor(0, W16 * 10);

  uint16_t label = 0;
  float max_y = 0.0f;

  noodle_find_max(output, 10, max_y, label);

  for (uint16_t i = 0; i < 10; i++)
  {
    tft.print(i);
    tft.print(" : ");
    tft.println(output->data[i], 4);
  }

  tft.print(F("\n\nPrediction: "));
  tft.print(label);

  tft.print(F(", in "));
  tft.print(et, 2);
  tft.println(F(" s."));
}

void loop()
{
  int16_t xpos, ypos;
  tp = ts.getPoint();

  pinMode(XM, OUTPUT);
  pinMode(YP, OUTPUT);

  if (tp.z > MINPRESSURE && tp.z < MAXPRESSURE)
  {
    xpos = map(tp.x, TS_LEFT, TS_RT, 0, 240);
    ypos = map(tp.y, TS_BOT, TS_TOP, 0, 320);

    if (((ypos - PENRADIUS) > 0) && ((ypos + PENRADIUS) < L) &&
        ((xpos - PENRADIUS) > 0) && ((xpos + PENRADIUS) < L))
    {
      tft.fillCircle(xpos, ypos, PENRADIUS, TFT_BLUE);
      track_roi(xpos, ypos);
    }

    if ((ypos > 320 - W8) && (xpos < W2))
    {
      reset_grid();
      area_setup();
    }

    if ((ypos > 320 - W8) && (xpos > W2))
    {
      for (uint16_t q = 0; q < 16 * 16; q++)
        GRID[q] = 0;

      int32_t sum_x = 0;
      int32_t sum_y = 0;
      int32_t count = 0;

      // Pass 1: compute center of mass of blue pixels
      for (byte i = 0; i < 16; i++)
      {
        for (byte j = 0; j < 16; j++)
        {
          for (byte k = 0; k < L16; k++)
          {
            for (byte l = 0; l < L16; l++)
            {
              int16_t x = i * L16 + k;
              int16_t y = j * L16 + l;

              uint16_t pixel = tft.readPixel(x, y);

              if (pixel == TFT_BLUE)
              {
                sum_x += x;
                sum_y += y;
                count++;
              }
            }
          }
        }
      }

      if (count == 0)
      {
        tft.println(F("No digit"));
        return;
      }

      int16_t cx = (int16_t)(sum_x / count);
      int16_t cy = (int16_t)(sum_y / count);

      float denom = (float)ROI[5];
      if (denom < 1.0f)
        denom = 1.0f;

      float target_content = (float)L * (20.0f / 28.0f);
      float s = target_content / denom;

      // Pass 2: remap blue pixels around center of mass
      for (byte i = 0; i < 16; i++)
      {
        for (byte j = 0; j < 16; j++)
        {
          for (byte k = 0; k < L16; k++)
          {
            for (byte l = 0; l < L16; l++)
            {
              int16_t x = i * L16 + k;
              int16_t y = j * L16 + l;

              uint16_t pixel = tft.readPixel(x, y);

              if (pixel == TFT_BLUE)
              {
                int16_t x_ = (int16_t)(s * (float)(x - cx) + (float)(L * 0.5f));
                int16_t y_ = (int16_t)(s * (float)(y - cy) + (float)(L * 0.5f));

                if ((x_ >= 0) && (x_ < L) && (y_ >= 0) && (y_ < L))
                {
                  uint8_t gx = x_ / L16;
                  uint8_t gy = y_ / L16;
                  uint16_t idx = gy * 16 + gx;

                  if (GRID[idx] < 255)
                    GRID[idx] = GRID[idx] + 1;
                }
              }
            }
          }
        }
      }

      normalize_grid();
      area_setup();

      tft.setTextSize(1);
      tft.setCursor(0, W16 * 7);

      unsigned long st = micros();

      FCNFile FCN1;
      FCN1.weight_fn = "w01.bin";
      FCN1.bias_fn = "b01.bin";
      FCN1.act = ACT_RELU;

      FCNFile FCN2;
      FCN2.weight_fn = "w02.bin";
      FCN2.bias_fn = "b02.bin";
      FCN2.act = ACT_SOFTMAX;

      tft.println(F("NN #1 ..."));
      uint16_t V = noodle_fcn(GRID, 256, 64, &A, FCN1, progress_hnd);

      if (V == 0)
      {
        tft.println(F("FCN1 failed"));
        return;
      }

      tft.println(F("NN #2 ..."));
      V = noodle_fcn(&A, V, 10, &B, FCN2, progress_hnd);

      if (V == 0)
      {
        tft.println(F("FCN2 failed"));
        return;
      }

      float et = (float)(micros() - st) * 1e-6;

      summarize(et, &B);
      reset_grid();
    }
  }
}
