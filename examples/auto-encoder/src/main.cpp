#include <Arduino.h>
#include "noodle.h"
#include "noodle_serial.h"

#include "w01.h"
#include "b01.h"
#include "w02.h"
#include "b02.h"
#include "w03.h"
#include "b03.h"
#include "w04.h"
#include "b04.h"

static constexpr uint16_t IMG_W = 28;
static constexpr uint16_t IMG_SIZE = IMG_W * IMG_W;

static uint8_t RX_BYTES[IMG_SIZE];
static uint8_t TX_BYTES[IMG_SIZE];

// Two grow-only tensor buffers for ping-pong inference.
static NoodleBuffer A;
static NoodleBuffer B;
static NoodleBuffer *Y = &A;

static inline void swap_buffers(NoodleBuffer *&x, NoodleBuffer *&y) {
  NoodleBuffer *t = x;
  x = y;
  y = t;
}

static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n) {
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i] * inv;
}

static void float_image_to_bytes(const float *src, uint8_t *dst, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float v = src[i];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    dst[i] = (uint8_t)(255.0f * v + 0.5f);
  }
}

static uint16_t run_autoencoder_forward() {
  ConvMem E1;
  E1.K = 3; E1.P = 65535; E1.S = 2; E1.OP = 0;
  E1.weight = w01; E1.bias = b01; E1.act = ACT_RELU;

  ConvMem E2;
  E2.K = 3; E2.P = 65535; E2.S = 2; E2.OP = 0;
  E2.weight = w02; E2.bias = b02; E2.act = ACT_RELU;

  ConvMem D1;
  D1.K = 3; D1.P = 65535; D1.S = 2; D1.OP = 1;
  D1.weight = w03; D1.bias = b03; D1.act = ACT_RELU;

  ConvMem D2;
  D2.K = 3; D2.P = 65535; D2.S = 2; D2.OP = 1;
  D2.weight = w04; D2.bias = b04; D2.act = ACT_NONE;

  Pool no_pool;
  no_pool.M = 1;
  no_pool.T = 1;

  NoodleBuffer *in = &A;
  NoodleBuffer *out = &B;
  uint16_t W = IMG_W;

  // X is in A.
  // Each layer writes to out, then we swap in/out.
  W = noodle_conv_float(in, 1, 16, out, W, E1, no_pool, NULL);       // A -> B
  swap_buffers(in, out);

  W = noodle_conv_float(in, 16, 32, out, W, E2, no_pool, NULL);      // B -> A
  swap_buffers(in, out);

  W = noodle_conv_transpose_float(in, 32, 16, out, W, D1, NULL);     // A -> B
  swap_buffers(in, out);

  W = noodle_conv_transpose_float(in, 16, 1, out, W, D2, NULL);      // B -> A
  swap_buffers(in, out);

  // Final output is now in 'in'.
  noodle_sigmoid(in->data, IMG_SIZE);
  Y = in;

  return W;
}

static void process_one_image() {
  // Put input image into A.
  bytes_to_float_image(RX_BYTES, noodle_buffer_require(&A, IMG_SIZE), IMG_SIZE);

  uint32_t t0 = micros();
  uint16_t W = run_autoencoder_forward();
  uint32_t dt = micros() - t0;

  if (W != IMG_W) {
    Serial.printf("ERR_BAD_W %u\n", W);
    NoodleSerial::print_ready();
    return;
  }

  float_image_to_bytes(Y->data, TX_BYTES, IMG_SIZE);
  NoodleSerial::send_output_image(TX_BYTES, IMG_SIZE, dt);
}

void setup() {
  NoodleSerial::begin();
  NoodleSerial::clear_input();

  Serial.println();
  Serial.println(F("BOOT SMART BUFFER AE"));

  noodle_buffer_init(&A);
  noodle_buffer_init(&B);

  NoodleSerial::print_ready();
}

void loop() {
  if (!NoodleSerial::wait_for_img_header()) {
    NoodleSerial::print_ready();
    delay(20);
    return;
  }

  if (!NoodleSerial::recv_image_chunked(RX_BYTES, IMG_SIZE)) {
    NoodleSerial::print_ready();
    return;
  }

  process_one_image();
}
