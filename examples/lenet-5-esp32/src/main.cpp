#include <Arduino.h>

#include "noodle.h"
#include "w01.h"
#include "b01.h"
#include "w02.h"
#include "b02.h"
#include "w03.h"
#include "b03.h"
#include "w04.h"
#include "b04.h"
#include "w05.h"
#include "b05.h"

// -----------------------------
// Serial RX protocol settings
// -----------------------------
static const uint32_t BAUD = 9600;
static const uint32_t RX_TIMEOUT_MS = 3000;   // timeout waiting for full frame
static const int IMG_W = 28;
static const int IMG_H = 28;
static const int IMG_SIZE = IMG_W * IMG_H;

// If your Python sender sends a 3-byte header "IMG" before the 784 bytes,
// uncomment this.
// #define USE_HEADER_IMG

// If you want input normalized to [0,1], uncomment this.
// #define NORMALIZE_0_1

// -----------------------------
// Buffers
// -----------------------------

float *GRID;
float *BUFFER1;
float *BUFFER3;

static uint8_t RX_BYTES[IMG_SIZE];

// Forward decl
static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms);
static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n);
static int argmax(const float *v, int n);

void alloc_buffers()
{
  // Input image as float (BUFFER1)
  GRID    = (float *)malloc(IMG_SIZE * sizeof(float));
  BUFFER1 = GRID;

  // Big feature map scratch (your original)
  BUFFER3 = (float *)malloc(14 * 14 * 6 * sizeof(float));

  if (!GRID || !BUFFER3) {
    Serial.println(F("ERROR: malloc failed (out of RAM)"));
    while (true) delay(1000);
  }
}

// Your original predict pipeline, but now it also prints a result
void predict()
{
  ConvMem cnn1;
  cnn1.K = 5;
  cnn1.P = 2;
  cnn1.S = 1; // same padding
  cnn1.weight = w01;
  cnn1.bias   = b01;

  ConvMem cnn2;
  cnn2.K = 5;
  cnn2.P = 0;
  cnn2.S = 1; // valid padding
  cnn2.weight = w02;
  cnn2.bias   = b02;

  Pool pool;
  pool.M = 2;
  pool.T = 2;

  FCNMem fcn_mem1;
  fcn_mem1.weight = w03;
  fcn_mem1.bias   = b03;
  fcn_mem1.act    = ACT_RELU;

  FCNMem fcn_mem2;
  fcn_mem2.weight = w04;
  fcn_mem2.bias   = b04;
  fcn_mem2.act    = ACT_RELU;

  FCNMem fcn_mem3;
  fcn_mem3.weight = w05;
  fcn_mem3.bias   = b05;
  fcn_mem3.act    = ACT_SOFTMAX;

  unsigned long st = micros();
  uint16_t V;

  // (Optional) keep these prints if you want verbose logs
  // Serial.println(INFO);

  V = noodle_conv_float(BUFFER1, 1, 6, BUFFER3, 28, cnn1, pool, NULL);
  V = noodle_conv_float(BUFFER3, 6, 16, BUFFER1, V, cnn2, pool, NULL);

  V = noodle_flat(BUFFER1, BUFFER3, V, 16);

  V = noodle_fcn(BUFFER3, V, 120, BUFFER1, fcn_mem1, NULL);
  V = noodle_fcn(BUFFER1, V, 84,  BUFFER3, fcn_mem2, NULL);
  V = noodle_fcn(BUFFER3, V, 10,  BUFFER1, fcn_mem3, NULL);
  
  float et = (float)(micros() - st) * 1e-6f;

  // BUFFER1 now holds 10 softmax 
  uint16_t pred;
  float max_val;
  noodle_find_max(BUFFER1, 10, max_val, pred);

  // Python-friendly single-line response:
  // PRED <digit> <seconds> <p0> <p1> ... <p9>
  Serial.print(F("PRED "));
  Serial.print(pred);
  Serial.print(' ');
  Serial.print(et, 4);
  Serial.print(' ');
  Serial.print(max_val, 4);
  Serial.println();
}

void setup()
{
  Serial.begin(BAUD);

  // Clear any garbage from boot
  delay(200);
  while (Serial.available()) Serial.read();

  //while (!noodle_fs_init()) {
  //  delay(500);
  //  Serial.println(F("."));
  // }
  //Serial.println(F("FFAT OK!"));

  alloc_buffers();

  // Tell Python we are alive
  Serial.println(F("READY"));
}

void loop()
{
  // 1) Receive 784 bytes
  if (!recv_exact(RX_BYTES, IMG_SIZE, RX_TIMEOUT_MS)) {
    Serial.println(F("READY"));
    return;
  }

  // 2) Convert to float image input
  bytes_to_float_image(RX_BYTES, BUFFER1, IMG_SIZE);

  // 3) Run inference + print result line
  predict();
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms)
{
  uint32_t t0 = millis();
  size_t got = 0;

  while (got < n) {
    if ((millis() - t0) > timeout_ms) return false;

    int avail = Serial.available();
    if (avail <= 0) { delay(1); continue; }

    int r = Serial.readBytes((char *)(dst + got), n - got);
    if (r > 0) got += (size_t)r;
  }
  return true;
}


static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n)
{
#ifdef NORMALIZE_0_1
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i] * inv;
#else
  // Keep “byte-ness” but stored as float: 0..255
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i];
#endif
}

static int argmax(const float *v, int n)
{
  int mi = 0;
  float mv = v[0];
  for (int i = 1; i < n; i++) {
    if (v[i] > mv) { mv = v[i]; mi = i; }
  }
  return mi;
}
