#include <Arduino.h>
#include "noodle.h"

#include <FS.h>
#include <FFat.h>

// -------- Model dims --------
static constexpr uint16_t INPUT_DIM = 640;
static constexpr uint16_t HIDDEN_DIM = 128;
static constexpr uint16_t BOTTLENECK_DIM = 8;

// Ping-pong buffers
static float BUF1[INPUT_DIM];
static float BUF2[INPUT_DIM];
static float X0[INPUT_DIM];

static float mse_640(const float* a, const float* b) {
  double acc = 0.0;
  for (int i = 0; i < INPUT_DIM; i++) {
    double d = (double)a[i] - (double)b[i];
    acc += d * d;
  }
  return (float)(acc / (double)INPUT_DIM);
}

static void run_aefc_forward() {
  FCNFile L1;  L1.weight_fn  = "/w01.txt"; L1.bias_fn  = "/w02.txt"; L1.act  = ACT_RELU;
  FCNFile L2;  L2.weight_fn  = "/w03.txt"; L2.bias_fn  = "/w04.txt"; L2.act  = ACT_RELU;
  FCNFile L3;  L3.weight_fn  = "/w05.txt"; L3.bias_fn  = "/w06.txt"; L3.act  = ACT_RELU;
  FCNFile L4;  L4.weight_fn  = "/w07.txt"; L4.bias_fn  = "/w08.txt"; L4.act  = ACT_RELU;
  FCNFile L5;  L5.weight_fn  = "/w09.txt"; L5.bias_fn  = "/w10.txt"; L5.act  = ACT_RELU;
  FCNFile L6;  L6.weight_fn  = "/w11.txt"; L6.bias_fn  = "/w12.txt"; L6.act  = ACT_RELU;
  FCNFile L7;  L7.weight_fn  = "/w13.txt"; L7.bias_fn  = "/w14.txt"; L7.act  = ACT_RELU;
  FCNFile L8;  L8.weight_fn  = "/w15.txt"; L8.bias_fn  = "/w16.txt"; L8.act  = ACT_RELU;
  FCNFile L9;  L9.weight_fn  = "/w17.txt"; L9.bias_fn  = "/w18.txt"; L9.act  = ACT_RELU;
  FCNFile L10; L10.weight_fn = "/w19.txt"; L10.bias_fn = "/w20.txt"; L10.act = ACT_NONE;

  uint16_t V = INPUT_DIM;

  V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L1,  NULL);
  V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L2,  NULL);
  V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L3,  NULL);
  V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L4,  NULL);
  V = noodle_fcn(BUF1, V, BOTTLENECK_DIM,  BUF2, L5,  NULL);
  V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L6,  NULL);
  V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L7,  NULL);
  V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L8,  NULL);
  V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L9,  NULL);
  V = noodle_fcn(BUF2, V, INPUT_DIM,       BUF1, L10, NULL);

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

  while (!noodle_sd_init())
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
