#include <Arduino.h>
#include "noodle.h"

#include <FS.h>
#include <FFat.h>

#include "w01.h"
#include "w02.h"
#include "w03.h"
#include "w04.h"
#include "w05.h"
#include "w06.h"
#include "w07.h"
#include "w08.h"
#include "w09.h"
#include "w10.h"
#include "w11.h"
#include "w12.h"
#include "w13.h"
#include "w14.h"
#include "w15.h"
#include "w16.h"
#include "w17.h"
#include "w18.h"
#include "w19.h"
#include "w20.h"

// -------- Model dims --------
static constexpr uint16_t INPUT_DIM = 640;
static constexpr uint16_t HIDDEN_DIM = 128;
static constexpr uint16_t BOTTLENECK_DIM = 8;

// Ping-pong buffers
static float BUF1[INPUT_DIM];
static float BUF2[INPUT_DIM];
static float X0[INPUT_DIM]; // We need to preserve input for later comparison

static float mse_640(const float* a, const float* b) {
  double acc = 0.0;
  for (int i = 0; i < INPUT_DIM; i++) {
    double d = (double)a[i] - (double)b[i];
    acc += d * d;
  }
  return (float)(acc / (double)INPUT_DIM);
}

static void run_aefc_forward() {
  FCNMem L1;  L1.weight  = w01; L1.bias  = w02; L1.act  = ACT_RELU;
  FCNMem L2;  L2.weight  = w03; L2.bias  = w04; L2.act  = ACT_RELU;
  FCNMem L3;  L3.weight  = w05; L3.bias  = w06; L3.act  = ACT_RELU;
  FCNMem L4;  L4.weight  = w07; L4.bias  = w08; L4.act  = ACT_RELU;
  FCNMem L5;  L5.weight  = w09; L5.bias  = w10; L5.act  = ACT_RELU;
  FCNMem L6;  L6.weight  = w11; L6.bias  = w12; L6.act  = ACT_RELU;
  FCNMem L7;  L7.weight  = w13; L7.bias  = w14; L7.act  = ACT_RELU;
  FCNMem L8;  L8.weight  = w15, L8.bias  = w16; L8.act  = ACT_RELU;
  FCNMem L9;  L9.weight  = w17; L9.bias  = w18; L9.act  = ACT_RELU;
  FCNMem L10; L10.weight = w19; L10.bias = w20; L10.act = ACT_NONE;

  uint16_t V = INPUT_DIM;

  V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L1,  NULL);
  //Serial.println(V);
  V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L2,  NULL);
  //Serial.println(V);
  V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L3,  NULL);
  //Serial.println(V);
  V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L4,  NULL);
  //Serial.println(V);
  V = noodle_fcn(BUF1, V, BOTTLENECK_DIM,  BUF2, L5,  NULL);
  //Serial.println(V);
  V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L6,  NULL);
  //Serial.println(V);
  V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L7,  NULL);
  //Serial.println(V);
  V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L8,  NULL);
  //Serial.println(V);
  V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L9,  NULL);
  //Serial.println(V);
  V = noodle_fcn(BUF2, V, INPUT_DIM,       BUF1, L10, NULL);
  //Serial.println(V);

  if (V != INPUT_DIM) Serial.printf("WARN bad_V=%u\n", V);
}

static float run_one_file(const char* path, uint32_t* out_us) {
  // Load input vector
  noodle_array_from_file(path, BUF1, INPUT_DIM);
  memcpy(X0, BUF1, sizeof(X0));

  uint32_t t0 = micros();
  run_aefc_forward();
  uint32_t dt = micros() - t0;

  if (out_us) *out_us = dt;
  return mse_640(X0, BUF1);
}

static void run_set(const char* prefix) {
  // prefix = "anom" or "norm"
  float mses[5];
  uint32_t us[5];

  Serial.printf("\n=== %s set ===\n", prefix);

  for (int i = 1; i <= 5; i++) {
    char path[32];
    snprintf(path, sizeof(path), "/%s%d.txt", prefix, i);

    mses[i-1] = run_one_file(path, &us[i-1]);

    Serial.printf("%s%d: mse=%.9g us=%lu\n",
                  prefix, i, mses[i-1], (unsigned long)us[i-1]);
    delay(10);
  }

  // quick summary
  double mean = 0.0;
  for (int i = 0; i < 5; i++) mean += (double)mses[i];
  mean /= 5.0;
  Serial.printf("Mean %s MSE = %.9g\n", prefix, (float)mean);
}

void setup() {
  Serial.begin(115200);
  delay(300);

  while (!noodle_fs_init())
  { // 4-bit
    delay(500);
    Serial.println(".");
  }
  Serial.println(F("FFAT OK!"));
}

void loop() {
  static bool done = false;
  if (done) { delay(1000); return; }

  run_set("anom");
  run_set("norm");

  done = true;
  Serial.println("\nDONE (processed anom1..5 + norm1..5)");
}
