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
static constexpr uint16_t IMG_H = 28;
static constexpr uint16_t IMG_SIZE = IMG_W * IMG_H;

// Optional: set to 1 for extra layer-shape prints. Keep 0 for Python plotting.
#ifndef AE_VERBOSE
#define AE_VERBOSE 0
#endif

static uint8_t RX_BYTES[IMG_SIZE];
static uint8_t TX_BYTES[IMG_SIZE];

// ============================================================
// Model dimensions
// ============================================================
// Keras:
//   input : 28 x 28 x 1
//   enc1  : Conv2D          1  -> 16, K=3, S=2, same, ReLU
//   enc2  : Conv2D          16 -> 32, K=3, S=2, same, ReLU
//   dec1  : Conv2DTranspose 32 -> 16, K=3, S=2, same, ReLU
//   dec2  : Conv2DTranspose 16 -> 1,  K=3, S=2, same, sigmoid
//
// Noodle tensor layout: CHW
//   X  : 1  x 28 x 28
//   Z1 : 16 x 14 x 14
//   Z2 : 32 x 7  x 7
//   Z3 : 16 x 14 x 14
//   Y  : 1  x 28 x 28
//
// This example uses memory-to-memory ping-pong inference:
//
//   BUF_A contains X
//   BUF_B contains Z1
//   BUF_A contains Z2
//   BUF_B contains Z3
//   BUF_A contains Y
//
// So the final denoised image is in BUF_A.
// ============================================================

static constexpr uint16_t W_IN  = 28;
static constexpr uint16_t C_IN  = 1;

static constexpr uint16_t W1    = 14;
static constexpr uint16_t C1    = 16;

static constexpr uint16_t W2    = 7;
static constexpr uint16_t C2    = 32;

static constexpr uint16_t W3    = 14;
static constexpr uint16_t C3    = 16;

static constexpr uint16_t W_OUT = 28;
static constexpr uint16_t C_OUT = 1;

// ============================================================
// Ping-pong tensor buffers and Noodle temp buffer
// ============================================================
// Permanent layer tensor storage:
//   Largest activation is 16 x 14 x 14 = 3136 floats.
//   Two buffers are enough for this straight-line network.
//
// Internal operator scratch storage:
//   Memory-to-memory Conv2D still requires temp_buff2.
//   Exact minimum for this model is 14 x 14 floats.
//   We use 28 x 28 floats here because it is beginner-safe and easy to explain.
// ============================================================

static constexpr size_t TENSOR_X_FLOATS  = (size_t)C_IN  * W_IN  * W_IN;
static constexpr size_t TENSOR_Z1_FLOATS = (size_t)C1    * W1    * W1;
static constexpr size_t TENSOR_Z2_FLOATS = (size_t)C2    * W2    * W2;
static constexpr size_t TENSOR_Z3_FLOATS = (size_t)C3    * W3    * W3;
static constexpr size_t TENSOR_Y_FLOATS  = (size_t)C_OUT * W_OUT * W_OUT;

static constexpr size_t PINGPONG_FLOATS = TENSOR_Z1_FLOATS;  // largest tensor: 3136 floats
static constexpr size_t TEMP2_FLOATS    = (size_t)IMG_W * IMG_W;

static_assert(PINGPONG_FLOATS >= TENSOR_X_FLOATS,  "BUF_A/BUF_B too small for X");
static_assert(PINGPONG_FLOATS >= TENSOR_Z1_FLOATS, "BUF_A/BUF_B too small for Z1");
static_assert(PINGPONG_FLOATS >= TENSOR_Z2_FLOATS, "BUF_A/BUF_B too small for Z2");
static_assert(PINGPONG_FLOATS >= TENSOR_Z3_FLOATS, "BUF_A/BUF_B too small for Z3");
static_assert(PINGPONG_FLOATS >= TENSOR_Y_FLOATS,  "BUF_A/BUF_B too small for Y");

static float *BUF_A = nullptr;
static float *BUF_B = nullptr;
static float *TEMP2 = nullptr;

// ============================================================
// Allocation
// ============================================================

static float *alloc_float_buffer(size_t n) {
#if defined(ARDUINO_ARCH_ESP32)
  float *p = (float *)ps_malloc(n * sizeof(float));
  if (p) return p;
#endif
  return (float *)malloc(n * sizeof(float));
}

static void alloc_buffers() {
  BUF_A = alloc_float_buffer(PINGPONG_FLOATS);
  BUF_B = alloc_float_buffer(PINGPONG_FLOATS);
  TEMP2 = alloc_float_buffer(TEMP2_FLOATS);

  if (!BUF_A || !BUF_B || !TEMP2) {
    Serial.println(F("ERROR allocation failed"));
    while (true) delay(1000);
  }

  // Required before using memory-to-memory noodle_conv_float().
  // BUF_A and BUF_B are layer tensor buffers.
  // TEMP2 is Noodle's internal Conv2D accumulation/pre-pooling scratch buffer.
  noodle_setup_temp_buffers(TEMP2);

  Serial.println(F("Ping-pong and temp buffers allocated"));
  Serial.printf("BUF_A floats=%u bytes=%u\n",
                (unsigned)PINGPONG_FLOATS,
                (unsigned)(PINGPONG_FLOATS * sizeof(float)));
  Serial.printf("BUF_B floats=%u bytes=%u\n",
                (unsigned)PINGPONG_FLOATS,
                (unsigned)(PINGPONG_FLOATS * sizeof(float)));
  Serial.printf("TEMP2 floats=%u bytes=%u\n",
                (unsigned)TEMP2_FLOATS,
                (unsigned)(TEMP2_FLOATS * sizeof(float)));
}

// ============================================================
// Utilities
// ============================================================

static void bytes_to_float_image_0_1(const uint8_t *src, float *dst, size_t n) {
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i] * inv;
}

static void float_image_to_bytes_0_255(const float *src, uint8_t *dst, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float v = src[i];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;

    int q = (int)(255.0f * v + 0.5f);
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    dst[i] = (uint8_t)q;
  }
}

static void print_stats(const char *name, const float *x, uint32_t n) {
  float mn = x[0];
  float mx = x[0];
  double sum = 0.0;

  for (uint32_t i = 0; i < n; i++) {
    const float v = x[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += (double)v;
  }

  Serial.printf("%s min=%.6f max=%.6f mean=%.6f\n",
                name, mn, mx, (float)(sum / (double)n));
}

// ============================================================
// Autoencoder forward, ping-pong memory-to-memory version
// ============================================================

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
  D2.weight = w04; D2.bias = b04;
  D2.act = ACT_NONE;  // ACT_SIGMOID is not in the current enum; sigmoid is applied below.

  Pool no_pool;
  no_pool.M = 1;
  no_pool.T = 1;

  float *A = BUF_A;
  float *B = BUF_B;

  uint16_t W = W_IN;

  // X is in A, Z1 will be in B.
  W = noodle_conv_float(A, C_IN, C1, B, W, E1, no_pool, NULL);
#if AE_VERBOSE
  Serial.printf("enc1 W=%u\n", W);
  print_stats("Z1", B, (uint32_t)C1 * W * W);
#endif
  if (W == 0) return 0;

  // Z1 is in B, Z2 will be in A.
  W = noodle_conv_float(B, C1, C2, A, W, E2, no_pool, NULL);
#if AE_VERBOSE
  Serial.printf("enc2 W=%u\n", W);
  print_stats("Z2", A, (uint32_t)C2 * W * W);
#endif
  if (W == 0) return 0;

  // Z2 is in A, Z3 will be in B.
  W = noodle_conv_transpose_float(A, C2, C3, B, W, D1, NULL);
#if AE_VERBOSE
  Serial.printf("dec1 W=%u\n", W);
  print_stats("Z3", B, (uint32_t)C3 * W * W);
#endif
  if (W == 0) return 0;

  // Z3 is in B, Y will be in A.
  W = noodle_conv_transpose_float(B, C3, C_OUT, A, W, D2, NULL);
#if AE_VERBOSE
  Serial.printf("dec2 W=%u\n", W);
  print_stats("Y pre-sigmoid", A, (uint32_t)C_OUT * W * W);
#endif
  if (W == 0) return 0;

  // Match Keras dec2 activation="sigmoid" while Noodle has no ACT_SIGMOID yet.
  // Final Y is in BUF_A.
  noodle_sigmoid(A, IMG_SIZE);

#if AE_VERBOSE
  print_stats("Y sigmoid", A, IMG_SIZE);
#endif

  return W;
}

static void process_one_image() {
  // RX_BYTES -> BUF_A.
  // After this point:
  //   BUF_A contains X.
  //   BUF_B is free for the first layer output.
  bytes_to_float_image_0_1(RX_BYTES, BUF_A, IMG_SIZE);

  uint32_t t0 = micros();
  uint16_t W_final = run_autoencoder_forward();
  uint32_t dt = micros() - t0;

  if (W_final != W_OUT) {
    Serial.printf("ERR_BAD_W %u\n", W_final);
    NoodleSerial::print_ready();
    return;
  }

  // Final Y is in BUF_A after the ping-pong forward pass.
  float_image_to_bytes_0_255(BUF_A, TX_BYTES, IMG_SIZE);

  NoodleSerial::send_output_image(TX_BYTES, IMG_SIZE, dt);
}

// ============================================================
// Arduino
// ============================================================

void setup() {
  NoodleSerial::begin();

  NoodleSerial::clear_input();

  Serial.println();
  Serial.println(F("BOOT OK"));

#if defined(ARDUINO_ARCH_ESP32)
  Serial.printf("Flash size: %u\n", ESP.getFlashChipSize());
  Serial.printf("PSRAM found: %s\n", psramFound() ? "YES" : "NO");
  Serial.printf("PSRAM size: %u\n", ESP.getPsramSize());
#endif

  alloc_buffers();
  NoodleSerial::print_ready();
}

void loop() {
  // Wait for a valid image header.
  //
  // Important:
  // Python may open the USB CDC port after the single setup() READY has already
  // been printed or flushed. Therefore, if no IMG header arrives within the
  // timeout, announce READY again. This makes the protocol recoverable without
  // pressing reset on the ESP32.
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
