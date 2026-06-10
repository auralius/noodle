// Last tested on ESP32-DEVKIT, June 10, 2026

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
static const uint32_t RX_TIMEOUT_MS = 3000;

static const uint16_t IMG_W = 28;
static const uint16_t IMG_H = 28;
static const uint16_t IMG_SIZE = IMG_W * IMG_H;

// If your Python sender sends a 3-byte header "IMG" before the 784 bytes,
// uncomment this.
// #define USE_HEADER_IMG

// If you want input normalized to [0,1], uncomment this.
// #define NORMALIZE_0_1

// -----------------------------
// Smart grow-only buffers
// -----------------------------
static NoodleBuffer A;
static NoodleBuffer B;

static uint8_t RX_BYTES[IMG_SIZE];

// Forward decl
static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms);
static void bytes_to_float_image(const uint8_t *src, NoodleBuffer *dst, size_t n);
static inline void swap_buffers(NoodleBuffer *&in, NoodleBuffer *&out);

void init_buffers()
{
  noodle_buffer_init(&A);
  noodle_buffer_init(&B);

  // Allocate the first input buffer once.
  // Later layers grow A/B automatically as needed.
  if (!noodle_buffer_require(&A, IMG_SIZE)) {
    Serial.println(F("ERROR: buffer allocation failed"));
    while (true) delay(1000);
  }
}

void predict()
{
  ConvMem cnn1{};
  cnn1.K = 5;
  cnn1.P = 2;
  cnn1.S = 1; // same padding
  cnn1.weight = w01;
  cnn1.bias   = b01;
  cnn1.act    = ACT_NONE;

  ConvMem cnn2{};
  cnn2.K = 5;
  cnn2.P = 0;
  cnn2.S = 1; // valid padding
  cnn2.weight = w02;
  cnn2.bias   = b02;
  cnn2.act    = ACT_NONE;

  Pool pool{};
  pool.M = 2;
  pool.T = 2;

  FCNMem fcn_mem1{};
  fcn_mem1.weight = w03;
  fcn_mem1.bias   = b03;
  fcn_mem1.act    = ACT_RELU;

  FCNMem fcn_mem2{};
  fcn_mem2.weight = w04;
  fcn_mem2.bias   = b04;
  fcn_mem2.act    = ACT_RELU;

  FCNMem fcn_mem3{};
  fcn_mem3.weight = w05;
  fcn_mem3.bias   = b05;
  fcn_mem3.act    = ACT_SOFTMAX;

  unsigned long st = micros();

  NoodleBuffer *in  = &A;
  NoodleBuffer *out = &B;

  uint16_t V = IMG_W;

  V = noodle_conv_float(in, 1, 6, out, V, cnn1, pool, NULL);
  swap_buffers(in, out);

  V = noodle_conv_float(in, 6, 16, out, V, cnn2, pool, NULL);
  swap_buffers(in, out);

  V = noodle_flat(in, out, V, 16);
  swap_buffers(in, out);

  V = noodle_fcn(in, V, 120, out, fcn_mem1, NULL);
  swap_buffers(in, out);

  V = noodle_fcn(in, V, 84, out, fcn_mem2, NULL);
  swap_buffers(in, out);

  V = noodle_fcn(in, V, 10, out, fcn_mem3, NULL);
  swap_buffers(in, out);

  float et = (float)(micros() - st) * 1e-6f;

  uint16_t pred;
  float max_val;
  noodle_find_max(in, 10, max_val, pred);

  // Python-friendly single-line response:
  // PRED <digit> <seconds> <p_max>
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

  init_buffers();

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

  // 2) Convert to float image input in buffer A
  bytes_to_float_image(RX_BYTES, &A, IMG_SIZE);

  // 3) Run inference + print result line
  predict();
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

static inline void swap_buffers(NoodleBuffer *&in, NoodleBuffer *&out)
{
  NoodleBuffer *tmp = in;
  in = out;
  out = tmp;
}

static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms)
{
  uint32_t t0 = millis();
  size_t got = 0;

  while (got < n) {
    if ((millis() - t0) > timeout_ms) return false;

    int avail = Serial.available();
    if (avail <= 0) {
      delay(1);
      continue;
    }

    int r = Serial.readBytes((char *)(dst + got), n - got);
    if (r > 0) got += (size_t)r;
  }

  return true;
}

static void bytes_to_float_image(const uint8_t *src, NoodleBuffer *dst, size_t n)
{
  float *x = noodle_buffer_require(dst, n);
  if (!x) return;

#ifdef NORMALIZE_0_1
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) x[i] = (float)src[i] * inv;
#else
  // Keep "byte-ness" but stored as float: 0..255
  for (size_t i = 0; i < n; i++) x[i] = (float)src[i];
#endif
}
