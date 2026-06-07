#include <Arduino.h>
#include <string.h>

#include "noodle.h"
#include "noodle_serial.h"

// ============================================================
// ECG500 1D CNN classifier for Noodle / ESP32
//
// Keras model:
//   Input(500,1)
//   Conv1D(8, 5, valid, ReLU)
//   MaxPooling1D(2)
//   Conv1D(16, 3, valid, ReLU)
//   MaxPooling1D(2)
//   GlobalMaxPooling1D()
//   Dense(16, ReLU)
//   Dropout(0.2)       // ignored during inference
//   Dense(4, Softmax)
//
// Tensor layout:
//   compact channel-first 1D: [C][W]
//
// Weight layout expected:
//   w01: Conv1D [O][I][K] = [8][1][5]
//   b01: [8]
//   w02: Conv1D [O][I][K] = [16][8][3]
//   b02: [16]
//   w03: Dense  [O][I]    = [16][16]
//   b03: [16]
//   w04: Dense  [O][I]    = [4][16]
//   b04: [4]
//
// Export from Keras:
//   Conv1D kernel [K][I][O] -> transpose to [O][I][K]
//   Dense kernel  [I][O]   -> transpose to [O][I]
//
// This version uses ping-pong buffers:
//
//   A = input X          [1][500]
//   B = P1              [8][248]
//   A = P2              [16][123]
//   A = GMP             [16]
//   B = FC1             [16]
//   A = scores/softmax  [4]
//
// Noodle's temp_buff2 is still used as one pre-pooling Conv1D channel.
// ============================================================

#include "w01.h"
#include "b01.h"
#include "w02.h"
#include "b02.h"
#include "w03.h"
#include "b03.h"
#include "w04.h"
#include "b04.h"

// ============================================================
// Model dimensions
// ============================================================

static constexpr uint16_t ECG_LEN = 500;
static constexpr uint16_t N_CLASSES = 4;

static constexpr uint16_t C_IN = 1;

static constexpr uint16_t C1 = 8;
static constexpr uint16_t K1 = 5;
static constexpr uint16_t W1_POOL = 248;   // Conv valid 496 -> MaxPool1D(2)

static constexpr uint16_t C2 = 16;
static constexpr uint16_t K2 = 3;
static constexpr uint16_t W2_POOL = 123;   // Conv valid 246 -> MaxPool1D(2)

static constexpr uint16_t FC1_IN = 16;     // GlobalMaxPooling1D output
static constexpr uint16_t FC1_OUT = 16;
static constexpr uint16_t FC2_OUT = 4;

static constexpr size_t RX_NBYTES = (size_t)ECG_LEN * sizeof(float);
static constexpr size_t TX_NBYTES = (size_t)N_CLASSES * sizeof(float);

// Ping-pong buffer sizes in floats.
static constexpr uint16_t BUF_A_LEN = (C2 * W2_POOL);  // max(500, 16*123, 16, 4) = 1968
static constexpr uint16_t BUF_B_LEN = (C1 * W1_POOL);  // max(8*248, 16) = 1984

// Internal-pooling Conv1D uses Noodle's temp_buff2 as one pre-pooling output channel.
// Largest pre-pooling Conv1D output here is 496, so 500 floats is safe.
static constexpr uint16_t TMP_CONV_LEN = ECG_LEN;

#ifndef ECG_VERBOSE
#define ECG_VERBOSE 0
#endif

// ============================================================
// Buffers
// ============================================================

static uint8_t RX_BYTES[RX_NBYTES];
static uint8_t TX_BYTES[TX_NBYTES];

static float *A        = nullptr;  // ping-pong A: X, P2, GMP, final scores
static float *B        = nullptr;  // ping-pong B: P1, FC1
static float *TMP_CONV = nullptr;  // Noodle temp_buff2: one pre-pooling Conv1D channel

// ============================================================
// Allocation
// ============================================================

static float *alloc_float_buffer(size_t n) {
#if defined(ARDUINO_ARCH_ESP32)
  if (psramFound()) {
    float *p = (float *)ps_malloc(n * sizeof(float));
    if (p) return p;
  }
#endif
  return (float *)malloc(n * sizeof(float));
}

static void alloc_buffers() {
  A        = alloc_float_buffer(BUF_A_LEN);
  B        = alloc_float_buffer(BUF_B_LEN);
  TMP_CONV = alloc_float_buffer(TMP_CONV_LEN);

  if (!A || !B || !TMP_CONV) {
    Serial.println(F("ERROR allocation failed"));
    while (true) delay(1000);
  }

  // Required by noodle_conv1d(..., ConvMem, Pool, ...):
  // it uses temp_buff2 as the one-channel pre-pooling accumulation buffer.
  noodle_setup_temp_buffers((void *)TMP_CONV);

  Serial.println(F("Buffers allocated"));
  Serial.printf("Buffers: A=%u B=%u TMP=%u floats\n",
                (unsigned)BUF_A_LEN,
                (unsigned)BUF_B_LEN,
                (unsigned)TMP_CONV_LEN);
}

// ============================================================
// Small helpers
// ============================================================

static void print_stats(const char *name, const float *x, uint32_t n) {
  if (!x || n == 0) return;

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

static void bytes_to_float_array(const uint8_t *src, float *dst, size_t n_float) {
  memcpy(dst, src, n_float * sizeof(float));
}

static void float_array_to_bytes(const float *src, uint8_t *dst, size_t n_float) {
  memcpy(dst, src, n_float * sizeof(float));
}

static uint16_t argmax4(const float *x) {
  uint16_t imax = 0;
  float vmax = x[0];

  for (uint16_t i = 1; i < N_CLASSES; i++) {
    if (x[i] > vmax) {
      vmax = x[i];
      imax = i;
    }
  }

  return imax;
}

// ============================================================
// ECG CNN forward
// ============================================================

static bool run_ecg_forward() {
  ConvMem conv1;
  conv1.K = K1;
  conv1.P = 0;          // Keras Conv1D default padding="valid"
  conv1.S = 1;
  conv1.weight = w01;
  conv1.bias = b01;
  conv1.act = ACT_RELU;

  ConvMem conv2;
  conv2.K = K2;
  conv2.P = 0;          // Keras Conv1D default padding="valid"
  conv2.S = 1;
  conv2.weight = w02;
  conv2.bias = b02;
  conv2.act = ACT_RELU;

  Pool pool;
  pool.M = 2;
  pool.T = 2;

  FCNMem fc1;
  fc1.weight = w03;
  fc1.bias = b03;
  fc1.act = ACT_RELU;

  FCNMem fc2;
  fc2.weight = w04;
  fc2.bias = b04;
  fc2.act = ACT_SOFTMAX;

  uint16_t W = ECG_LEN;

  // A: X[1][500] -> B: P1[8][248]
  W = noodle_conv1d(A, C_IN, B, C1, W, conv1, pool, NULL);
  if (W != W1_POOL) {
    Serial.printf("ERR_BAD_W_POOL1 %u\n", W);
    return false;
  }

#if ECG_VERBOSE
  print_stats("P1/B", B, (uint32_t)C1 * W1_POOL);
#endif

  // B: P1[8][248] -> A: P2[16][123]
  W = noodle_conv1d(B, C1, A, C2, W, conv2, pool, NULL);
  if (W != W2_POOL) {
    Serial.printf("ERR_BAD_W_POOL2 %u\n", W);
    return false;
  }

#if ECG_VERBOSE
  print_stats("P2/A", A, (uint32_t)C2 * W2_POOL);
#endif

  // A: P2[16][123] -> A[0:16]: GMP[16]
  if (noodle_gmp(A, C2, W) != FC1_IN) {
    Serial.println(F("ERR_GMP"));
    return false;
  }

#if ECG_VERBOSE
  print_stats("GMP/A", A, FC1_IN);
#endif

  // A: GMP[16] -> B: FC1[16]
  if (noodle_fcn(A, FC1_IN, FC1_OUT, B, fc1, NULL) != FC1_OUT) {
    Serial.println(F("ERR_FC1"));
    return false;
  }

#if ECG_VERBOSE
  print_stats("FC1/B", B, FC1_OUT);
#endif

  // B: FC1[16] -> A: scores[4]
  if (noodle_fcn(B, FC1_OUT, FC2_OUT, A, fc2, NULL) != FC2_OUT) {
    Serial.println(F("ERR_FC2"));
    return false;
  }

#if ECG_VERBOSE
  Serial.printf("PRED %u probs=%.6f %.6f %.6f %.6f\n",
                argmax4(A), A[0], A[1], A[2], A[3]);
#endif

  return true;
}

static void process_one_ecg() {
  // Host sends 500 little-endian float32 values into A.
  bytes_to_float_array(RX_BYTES, A, ECG_LEN);

  const uint32_t t0 = micros();
  const bool ok = run_ecg_forward();
  const uint32_t dt = micros() - t0;

  if (!ok) {
    NoodleSerial::print_ready();
    return;
  }

  // Return 4 float32 probabilities: [Brady, Normal, Tachy, Irregular].
  float_array_to_bytes(A, TX_BYTES, N_CLASSES);
  NoodleSerial::send_output_image(TX_BYTES, TX_NBYTES, dt);
}

// ============================================================
// Arduino
// ============================================================

void setup() {
  NoodleSerial::begin(115200);
  NoodleSerial::clear_input();

  Serial.println();
  Serial.println(F("BOOT ECG500 CNN PINGPONG OK"));

#if defined(ARDUINO_ARCH_ESP32)
  Serial.printf("Flash size: %u\n", ESP.getFlashChipSize());
  Serial.printf("PSRAM found: %s\n", psramFound() ? "YES" : "NO");
  Serial.printf("PSRAM size: %u\n", ESP.getPsramSize());
#endif

  alloc_buffers();
  NoodleSerial::print_ready();
}

void loop() {
  // Protocol:
  //   MCU prints READY
  //   Host sends header "IMG"
  //   MCU prints RDYIMG
  //   Host sends 500 float32 samples in chunks and waits for ACK per chunk
  //   MCU prints OUT <dt_us>, sends 4 float32 probabilities, then READY
  if (!NoodleSerial::wait_for_img_header()) {
    NoodleSerial::print_ready();
    delay(20);
    return;
  }

  if (!NoodleSerial::recv_image_chunked(RX_BYTES, RX_NBYTES)) {
    NoodleSerial::print_ready();
    return;
  }

  process_one_ecg();
}
