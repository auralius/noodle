/**
 * Auralius Manurung -- Universitas Telkom, Bandung
 *   auralius.manurung@ieee.org
 * Lisa Kristiana -- ITENAS, Bandung
 *   lisa@itenas.ac.id
 */
#include <MCUFRIEND_kbv.h>
MCUFRIEND_kbv tft;  // hard-wiTFT_RED for UNO shields anyway.


#include <TouchScreen.h>
const uint16_t XP = 6, XM = A2, YP = A1, YM = 7;  //240x320 ID=0x9341
//const int XP = 8, YP = A3, XM = A2, YM = 9;
const int16_t TS_LEFT = 154, TS_RT = 902, TS_TOP = 184, TS_BOT = 919;
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);
TSPoint tp;


#include "noodle.h"
#include "bmp.h"

#define SD_CS 10
#define MINPRESSURE 200
#define MAXPRESSURE 1000
#define PENRADIUS 1


const int16_t L = 96;
const int16_t L8 = 96 / 8;
const int16_t L16 = 96 / 16;

const int16_t W8 = 240 / 8;
const int16_t W16 = 240 / 16;
const int16_t W2 = 240 / 2;

// Writing area
byte ROI[8];  //xmin, ymin, xmax, ymax, width, length, center x, center y

// The discretized drawing area: 16x16 grids, max value of each grid is 255
byte GRID[16 * 16];
float *OUTPUT_BUFFER;

// Grayscale colors in 17 steps: 0 to 16
uint16_t GREYS[17];


void progress_hnd(float p){
  if (p < 0.01) 
    tft.fillRect(0, L + 1, 240, 3, TFT_BLACK);
  else
    tft.fillRect(0, L + 1, p * 240.0, 3, TFT_BLUE);
}


// Grayscale of 16 steps from TFT_BLACK (0x0000) to TFT_GREEN(0xFFFF)
void create_greys() {
  for (int16_t k = 0; k < 16; k++)
    GREYS[k] = ((2 * k << 11) | (4 * k << 5) | (2 * k));
  GREYS[16] = 0xFFFF;
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


// Normalize the numbers in the grids such that they range from 0 to 16
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
    for (int16_t j = 0; j < 16; j++)
      tft.fillRect(i * L16, j * L16, L16, L16, GREYS[GRID[j * 16 + i] / 16]);

  tft.drawRect(0, 0, L, L, TFT_GREEN);

  tft.setTextSize(1);
  tft.setCursor(W2, 10);
  tft.print(F("LeNet-1 on UNO R4\n"));

  showBMP("logo.bmp", W2, 25);
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

  ROI[6] = ROI[0] + ROI[4] / 2;
  ROI[7] = ROI[1] + ROI[5] / 2;
}


// Draw the ROI as a TFT_RED rectangle
void draw_roi() {
  tft.drawRect(ROI[0], ROI[1], ROI[4], ROI[5], TFT_RED);
  tft.drawCircle(ROI[6], ROI[7], 2, TFT_RED);
}


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
  Serial.begin(9600);
  //while ( !Serial ) delay(2);

  create_greys();

  tft.reset();
  tft.begin(tft.readID());
  tft.fillScreen(TFT_BLACK);

  tft.setCursor(0, 10);
  tft.setTextSize(1);
  tft.setTextColor(TFT_GREEN);


  if (!SD.begin(SD_CS, 24000000)) {
    tft.println(F("cannot start SD"));
    while (1)
      ;
  }

  reset_grid();
  area_setup();
  draw_buttons("CLEAR", "PREDICT");

  OUTPUT_BUFFER = noodle_create_buffer(16 * 16);
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
    xpos = map(tp.x, TS_LEFT, TS_RT, 0, tft.width());
    ypos = map(tp.y, TS_BOT, TS_TOP, 0, tft.height());

    // are we in drawing area ?
    if (((ypos - PENRADIUS) > 0) && ((ypos + PENRADIUS) < L) && ((xpos - PENRADIUS) > 0) && ((xpos + PENRADIUS) < L)) {
      tft.fillCircle(xpos, ypos, PENRADIUS, TFT_BLUE);
      track_roi(xpos, ypos);
    }

    // CLEAR?
    if ((ypos > tft.height() - W8) && (xpos < W2)) {
      reset_grid();
      area_setup();
    }

    // PREDICT?
    if ((ypos > tft.height() - W8) && (xpos > W2)) {
      draw_roi();

      for (byte i = 0; i < 16; i++) {
        for (byte j = 0; j < 16; j++) {
          for (byte k = 0; k < L16; k++) {
            for (byte l = 0; l < L16; l++) {
              int16_t x = i * L16 + k;
              int16_t y = j * L16 + l;

              uint16_t pixel = tft.readPixel(x, y);

              if (pixel == TFT_BLUE) {
                float a = (float)L;
                float s = a / (float)ROI[5];
                int16_t x_ = s * (float)(x - ROI[0]) + 0.5 * (a - s * (float)ROI[4]);  // Align to center (60,60)
                int16_t y_ = s * (float)(y - ROI[1]) + 0.5 * (a - s * (float)ROI[5]);

                if ((x_ >= 0) && (x_ < L) && (y_ >= 0) && (y_ < L)) {
                  tft.fillCircle(x, y, 1, TFT_RED);
                  GRID[y_ / L16 * 16 + (x_ / L16)] = GRID[y_ / L16 * 16 + (x_ / L16)] + 1;
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

      noodle_grid_to_file(GRID, "i1-a.txt", 16);

      unsigned long st = micros();  // timer starts
      uint16_t V;

      tft.println(F("Conv layer 1 ..."));
      V = noodle_conv(GRID, OUTPUT_BUFFER, 1, 12, "i1-x.txt", "o1-x.txt", "w1-x-x.txt", "w2.txt", 16, 2, 5, 1, 2, 2, progress_hnd);

      tft.println(F("Conv layer 2 ..."));
      V = noodle_conv(GRID, OUTPUT_BUFFER, 12, 12, "o1-x.txt", "o2-x.txt", "w3-x-x.txt", "w4.txt", V, 2, 5, 1, 2, 2, progress_hnd);

      V = noodle_flat(OUTPUT_BUFFER, "o2-x.txt", V, 12);

      tft.println(F("NN layer 1 ..."));
      V = noodle_fcn(OUTPUT_BUFFER, V, 30, "o3.txt", "w5.txt", "w6.txt", progress_hnd);

      tft.println(F("NN layer 2 ..."));
      V = noodle_fcn("o3.txt", V, 10, OUTPUT_BUFFER, "w7.txt", "w8.txt", progress_hnd);

      float et = (float)(micros() - st) * 1e-6;  // timer stops

      summarize(et, OUTPUT_BUFFER);
      reset_grid();
    }
  }
}


void summarize(float et, float *output_buffer) {
  float *y = output_buffer;  // reuse global memory
  tft.setTextSize(1);
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
    if (i == label)
      tft.setTextColor(TFT_RED);

    tft.print(i);
    tft.print(" : ");
    tft.println(y[i], 4);

    if (i == label)
      tft.setTextColor(TFT_GREEN);
  }

  tft.print(F("\n----------------------------------\nPrediction: "));
  tft.setTextColor(TFT_RED);
  tft.print(label);
  tft.setTextColor(TFT_GREEN);

  // Display the elapsed time
  tft.print(F(" -- in "));
  tft.print(et, 2);
  tft.println(F(" seconds."));
}
