/**
 * Auralius Manurung -- Universitas Telkom, Bandung
 *   auralius.manurung@ieee.org
 * Lisa Kristiana -- ITENAS, Bandung
 *   lisa@itenas.ac.id
 */
#include <Arduino.h>
#include <MCUFRIEND_kbv.h>
#include <TouchScreen.h>
#include "noodle.h"

MCUFRIEND_kbv tft; 
// Typically there are 2 variant pin configurations for the touchscreen
const uint16_t XP = 6, XM = A2,YP = A1, YM = 7;     // variant 1
//const uint16_t XP = 8, YP = A3, XM = A2, YM = 9;  // variant 2
// Touch screen boundaries, obtained by calibration
const int TS_LEFT = 121, TS_RT = 906, TS_TOP = 944, TS_BOT=129;
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
byte ROI[8];  //xmin, ymin, xmax, ymax, width, length, center x, center y

// The discretized drawing area: 16x16 grids, max value of each grid is 255
byte GRID[16 * 16];
float OUTPUT_BUFFER1[64] = {0};
float OUTPUT_BUFFER2[10] = {0};

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

static inline uint16_t grey(uint8_t i) {
  return pgm_read_word(&GREYS[i]);
}

// Clear the grid data, set all to 0
void reset_grid() {
  for (int16_t i = 0; i < 16; i++)
    for (int16_t j = 0; j < 16; j++)
      GRID[i * 16 + j] = 0;

  // Reset ROI
  for (int16_t i = 0; i < 8; i++)
    ROI[i] = 0;

  ROI[0] = 250;  // xmin
  ROI[1] = 250;  // ymin
}


// Normalize the numbers in the grids such that they range from 0 to 255
void normalize_grid() {
  // find the maximum
  int16_t maxval = 0;
  for (int16_t i = 0; i < 16; i++)
    for (int16_t j = 0; j < 16; j++)
      if (GRID[i * 16 + j] > maxval)
        maxval = GRID[i * 16 + j];

  // normalize such that the maximum is 255.0
  for (int16_t i = 0; i < 16; i++)
    for (int16_t j = 0; j < 16; j++)
      GRID[i * 16 + j] = round((float)GRID[i * 16 + j] / (float)maxval * 255.0);  // round instead of floor!
}

// Draw the grids in the screen
// The fill-color or the gray-level of each grid is set based on its value
void area_setup() {
  tft.fillRect(0, 0, 240, 320 - W8, TFT_BLACK);

  for (int16_t i = 0; i < 16; i++)
    for (int16_t j = 0; j < 16; j++){
      uint8_t idx = GRID[j * 16 + i] >> 4;   // same as /16 for uint8
      if (idx > 16) idx = 16;
      tft.fillRect(i * L16, j * L16, L16, L16, grey(idx));
    }

  tft.drawRect(0, 0, L, L, TFT_GREEN);

  tft.setTextSize(1);
  tft.setCursor(W2, 10);
  tft.print(F("2-layer FCN UNO R3\n"));
}


// Track ROI, the specific area where the drawing occurs
void track_roi(int16_t xpos, int16_t ypos) {
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

  ROI[6] = ROI[0] + ROI[4] / 2; // center x
  ROI[7] = ROI[1] + ROI[5] / 2; // center y
}


// Draw the ROI as a TFT_RED rectangle
//void draw_roi() {
//  tft.drawRect(ROI[0], ROI[1], ROI[4], ROI[5], TFT_RED);
//  tft.drawCircle(ROI[6], ROI[7], 2, TFT_RED);
//}


// Draw the two button at the bottom of the screen
void draw_buttons(char *label1, char *label2) {
  // Add 2 buttons in the bottom: CLEAR and PREDICT
  tft.fillRect(0, 320 - W8, W2, W2, TFT_BLUE);
  tft.fillRect(W2, 320 - W8, W2, W2, TFT_RED);
  tft.setTextSize(2);
  tft.setCursor(6, 320 - W8 + 6);
  tft.print(label1);
  tft.setCursor(W2 + 8, 320 - W8 + 6);
  tft.print(label2);
}


// Arduino setup function
void setup(void) {
  //Serial.begin(115200);
  //while ( !Serial ) delay(2);

  tft.reset();
  tft.begin(tft.readID());
  
  tft.fillScreen(TFT_BLACK);

  tft.setCursor(0, 10);
  tft.setTextSize(1);
  tft.setTextColor(TFT_GREEN);

  if (!noodle_sd_init(SD_CS)) {
    tft.println(F("cannot start SD"));
    while (1)
      ;
  }
  

  reset_grid();
  area_setup();
  draw_buttons("CLEAR", "PREDICT");
}


void progress_hnd(float p){
  if (p < 0.01) 
    tft.fillRect(0, L + 1, 240, 3, TFT_BLACK);
  else
    tft.fillRect(0, L + 1, p * 240.0, 3, TFT_BLUE);
}

void summarize(float et, float *output_buffer) {
  float *y = output_buffer;  // reuse global memory
  //tft.setTextSize(1);
  tft.setCursor(0, W16 * 10);

  // Find the largest class
  uint16_t label = 0;
  float max_y = 0;

  for (uint16_t i = 0; i < 10; i++) {
    if (y[i] > max_y) {
      max_y = y[i];
      label = i;
    }
  }

  // Print the results
  for (uint16_t i = 0; i < 10; i++) {
    tft.print(i);
    tft.print(" : ");
    tft.println(y[i], 4);
  }

  tft.print(F("\n\nPrediction: "));
  tft.print(label);

  // Display the elapsed time
  tft.print(F(", in "));
  tft.print(et, 2);
  tft.println(F(" s."));
}

void loop() {
  int16_t xpos, ypos;  //screen coordinates
  tp = ts.getPoint();  //tp.x, tp.y are ADC values

  // if sharing pins, you'll need to fix the directions of the touchscreen pins
  pinMode(XM, OUTPUT);
  pinMode(YP, OUTPUT);


  // we have some minimum pressure we consider 'valid'
  if (tp.z > MINPRESSURE && tp.z < MAXPRESSURE) {
    /// Map to your current pixel orientation
    xpos = map(tp.x, TS_LEFT, TS_RT, 0, 240);
    ypos = map(tp.y, TS_TOP, TS_BOT, 0, 320);

    // are we in drawing area ?
    if (((ypos - PENRADIUS) > 0) && ((ypos + PENRADIUS) < L) && ((xpos - PENRADIUS) > 0) && ((xpos + PENRADIUS) < L)) {
      tft.fillCircle(xpos, ypos, PENRADIUS, TFT_BLUE);
      track_roi(xpos, ypos);
    }

    // CLEAR?
    if ((ypos > 320 - W8) && (xpos < W2)) {
      reset_grid();
      area_setup();
    }

    // PREDICT?
    if ((ypos > 320 - W8) && (xpos > W2)) {
      for (byte i = 0; i < 16; i++) {
        for (byte j = 0; j < 16; j++) {
          for (byte k = 0; k < L16; k++) {
            for (byte l = 0; l < L16; l++) {
              int16_t x = i * L16 + k;
              int16_t y = j * L16 + l;

              uint16_t pixel = tft.readPixel(x, y);

              if (pixel == TFT_BLUE) {
                float s = (float)(L-2) / (float)ROI[5];
                int16_t x_ = (int16_t)((x - ROI[6]) * s + (float)(L * 0.5f));
                int16_t y_ = (int16_t)((y - ROI[7]) * s + (float)(L * 0.5f));

                if ((x_ >= 0) && (x_ < L) && (y_ >= 0) && (y_ < L)) {
                  //tft.fillCircle(x, y, 1, TFT_RED);
                  GRID[y_ / L16 * 16 + (x_ / L16)] = GRID[y_ / L16 * 16 + (x_ / L16)] + 1;
                }
              }
            }
          }
        }
      }

      //draw_roi();
      normalize_grid();
      area_setup();

      tft.setTextSize(1);
      tft.setCursor(0, W16 * 7);

      unsigned long st = micros();  // timer starts

      FCNFile FCN1;
      FCN1.weight_fn = "w01.txt";
      FCN1.bias_fn = "b01.txt";
      FCN1.act = ACT_RELU;

      FCNFile FCN2;
      FCN2.weight_fn = "w02.txt";
      FCN2.bias_fn = "b02.txt";
      FCN2.act = ACT_SOFTMAX;

      tft.println(F("NN #1 ..."));
      // 256 input neurons, 64 hidden neurons
      uint16_t V = noodle_fcn(GRID, 256, 64, OUTPUT_BUFFER1, FCN1, progress_hnd);

      tft.println(F("NN #2 ..."));
      // 10 output neurons
      V = noodle_fcn(OUTPUT_BUFFER1, V, 10, OUTPUT_BUFFER2, FCN2, progress_hnd);

      float et = (float)(micros() - st) * 1e-6;  // timer stops

      summarize(et, OUTPUT_BUFFER2);
      reset_grid();
    }
  }
}

