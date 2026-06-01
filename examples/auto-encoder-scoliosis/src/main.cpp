#include <Arduino.h>
#include "noodle.h"
#include "noodle_serial.h"

// Exported by model_exporter_convtranspose.py from autoencoder_revised.tflite.
// Order:
//   w01/b01 Conv2D           3 -> 32
//   w02/b02 Conv2D          32 -> 64
//   w03/b03 Conv2D          64 -> 128
//   w04/b04 Dense        32768 -> 64
//   w05/b05 Dense           64 -> 32768
//   w06/b06 Conv2DTranspose 128 -> 128
//   w07/b07 Conv2DTranspose 128 -> 64
//   w08/b08 Conv2DTranspose  64 -> 32
//   w09/b09 Conv2DTranspose  32 -> 3
#include "w01.h"
#include "b01.h"
#include "w02.h"
#include "b02.h"
#include "w03.h"
#include "b03.h"
//#include "w04.h"
//#include "b04.h"
//#include "w05.h"
//#include "b05.h"
#include "w06.h"
#include "b06.h"
#include "w07.h"
#include "b07.h"
#include "w08.h"
#include "b08.h"
#include "w09.h"
#include "b09.h"

// ============================================================
// Serial payload
// ============================================================
// Protocol is intentionally the same style as the working MNIST denoising demo:
//
// PC -> MCU:
//   "IMG"
//   128*128*3 uint8 bytes in RGB HWC order, sent in chunks.
//   Firmware replies:
//      RDYIMG
//      ACK after every chunk
//
// MCU -> PC:
//   OUT <microseconds>\n
//   128*128*3 uint8 bytes in RGB HWC order
//   \n
//   READY
// ============================================================

static constexpr uint16_t IMG_W = 128;
static constexpr uint16_t IMG_H = 128;
static constexpr uint16_t IMG_C = 3;
static constexpr uint32_t IMG_SIZE = (uint32_t)IMG_W * IMG_H * IMG_C;

// Optional: set to 1 for layer shape/stats prints.
// Keep 0 when using Python binary receiver/plotter.
#ifndef AE_VERBOSE
#define AE_VERBOSE 1
#endif

static uint8_t *RX_BYTES = nullptr;   // RGB HWC, uint8
static uint8_t *TX_BYTES = nullptr;   // RGB HWC, uint8

// ============================================================
// Model dimensions
// ============================================================
// Keras/TFLite:
//   input                  : 128 x 128 x 3
//   conv2d                 : 128 x 128 x 32, then maxpool -> 64 x 64 x 32
//   conv2d_1               :  64 x  64 x 64, then maxpool -> 32 x 32 x 64
//   conv2d_2               :  32 x  32 x128, then maxpool -> 16 x 16 x128
//   flatten                : 32768
//   dense latent           : 64
//   dense expand           : 32768
//   reshape                : 16 x 16 x128
//   conv2d_transpose       : 32 x 32 x128
//   conv2d_transpose_1     : 64 x 64 x 64
//   conv2d_transpose_2     :128 x128 x 32
//   output transpose conv  :128 x128 x  3, then sigmoid
//
// Noodle internal tensor layout: CHW.
// Serial image layout: HWC RGB.
// ============================================================

static constexpr uint16_t W_IN  = 128;
static constexpr uint16_t C_IN  = 3;

static constexpr uint16_t W1_PRE = 128;
static constexpr uint16_t W1     = 64;
static constexpr uint16_t C1     = 32;

static constexpr uint16_t W2_PRE = 64;
static constexpr uint16_t W2     = 32;
static constexpr uint16_t C2     = 64;

static constexpr uint16_t W3_PRE = 32;
static constexpr uint16_t W3     = 16;
static constexpr uint16_t C3     = 128;

static constexpr uint16_t LATENT = 64;
static constexpr uint16_t FLAT_N = 16 * 16 * 128;  // 32768

static constexpr uint16_t W4 = 32;
static constexpr uint16_t C4 = 128;

static constexpr uint16_t W5 = 64;
static constexpr uint16_t C5 = 64;

static constexpr uint16_t W6 = 128;
static constexpr uint16_t C6 = 32;

static constexpr uint16_t W_OUT = 128;
static constexpr uint16_t C_OUT = 3;

// Encoder buffers are intentionally allocated large enough to hold the
// pre-pooling convolution outputs. After noodle_valid_max_pool(), the pooled
// tensor is compacted at the beginning of the same buffer.
static float *X     = nullptr;  // [3][128][128]
static float *Z1    = nullptr;  // capacity [32][128][128], after pool [32][64][64]
static float *Z2    = nullptr;  // capacity [64][64][64],  after pool [64][32][32]
static float *Z3    = nullptr;  // capacity [128][32][32], after pool [128][16][16]

static float *FLAT  = nullptr;  // HWC flatten, 32768
static float *LAT   = nullptr;  // 64
static float *DENSE = nullptr;  // HWC dense output, 32768
static float *R3    = nullptr;  // [128][16][16] CHW reshape

static float *D1    = nullptr;  // [128][32][32]
static float *D2    = nullptr;  // [64][64][64]
static float *D3    = nullptr;  // [32][128][128]
static float *Y     = nullptr;  // [3][128][128]

static float *TEMP2 = nullptr;  // one pre-pooling convolution plane

// ============================================================
// Allocation
// ============================================================

static void *alloc_bytes(size_t n) {
#if defined(ARDUINO_ARCH_ESP32)
  void *p = ps_malloc(n);
  if (p) return p;
#endif
  return malloc(n);
}

static float *alloc_float_buffer(size_t n) {
  return (float *)alloc_bytes(n * sizeof(float));
}

static void alloc_buffers() {
  RX_BYTES = (uint8_t *)alloc_bytes(IMG_SIZE);
  TX_BYTES = (uint8_t *)alloc_bytes(IMG_SIZE);

  X     = alloc_float_buffer((size_t)C_IN  * W_IN    * W_IN);
  Z1    = alloc_float_buffer((size_t)C1 * W1 * W1);  // 32 * 64 * 64
  Z2    = alloc_float_buffer((size_t)C2 * W2 * W2);  // 64 * 32 * 32
  Z3    = alloc_float_buffer((size_t)C3 * W3 * W3);  // 128 * 16 * 16

  FLAT  = alloc_float_buffer(FLAT_N);
  LAT   = alloc_float_buffer(LATENT);
  DENSE = alloc_float_buffer(FLAT_N);
  R3    = alloc_float_buffer((size_t)C3    * W3      * W3);

  D1    = alloc_float_buffer((size_t)C4    * W4      * W4);
  D2    = alloc_float_buffer((size_t)C5    * W5      * W5);
  D3    = alloc_float_buffer((size_t)C6    * W6      * W6);
  Y     = alloc_float_buffer((size_t)C_OUT * W_OUT   * W_OUT);

  TEMP2 = alloc_float_buffer((size_t)W_IN * W_IN);

  if (!RX_BYTES || !TX_BYTES ||
      !X || !Z1 || !Z2 || !Z3 || !FLAT || !LAT || !DENSE || !R3 ||
      !D1 || !D2 || !D3 || !Y || !TEMP2) {
    Serial.println(F("ERROR allocation failed"));
    while (true) delay(1000);
  }

  Serial.println(F("Buffers allocated"));
}

// ============================================================
// Utilities
// ============================================================

static void rgb888_hwc_to_chw_float_0_1(const uint8_t *src, float *dst, uint16_t W) {
  const float inv = 1.0f / 255.0f;
  const uint32_t plane = (uint32_t)W * W;

  for (uint16_t y = 0; y < W; y++) {
    for (uint16_t x = 0; x < W; x++) {
      const uint32_t pix = (uint32_t)y * W + x;
      dst[0 * plane + pix] = (float)src[pix * 3 + 0] * inv;
      dst[1 * plane + pix] = (float)src[pix * 3 + 1] * inv;
      dst[2 * plane + pix] = (float)src[pix * 3 + 2] * inv;
    }
  }
}

static void chw_float_0_1_to_rgb888_hwc(const float *src, uint8_t *dst, uint16_t W) {
  const uint32_t plane = (uint32_t)W * W;

  for (uint16_t y = 0; y < W; y++) {
    for (uint16_t x = 0; x < W; x++) {
      const uint32_t pix = (uint32_t)y * W + x;

      for (uint16_t c = 0; c < 3; c++) {
        float v = src[(uint32_t)c * plane + pix];

        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;

        int q = (int)(255.0f * v + 0.5f);
        if (q < 0) q = 0;
        if (q > 255) q = 255;

        dst[pix * 3 + c] = (uint8_t)q;
      }
    }
  }
}

#if AE_VERBOSE
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
#endif

// ============================================================
// Autoencoder forward
// ============================================================

static uint16_t run_autoencoder_forward() {
  Pool no_pool;
  no_pool.M = 1;
  no_pool.T = 1;

  Pool pool2;
  pool2.M = 2;
  pool2.T = 2;

  ConvMem E1;
  E1.K = 3; E1.P = 65535; E1.S = 1; E1.OP = 0;
  E1.weight = w01; E1.bias = b01; E1.act = ACT_RELU;

  ConvMem E2;
  E2.K = 3; E2.P = 65535; E2.S = 1; E2.OP = 0;
  E2.weight = w02; E2.bias = b02; E2.act = ACT_RELU;

  ConvMem E3;
  E3.K = 3; E3.P = 65535; E3.S = 1; E3.OP = 0;
  E3.weight = w03; E3.bias = b03; E3.act = ACT_RELU;

  FCNFile F1;
  F1.weight_fn = "/w04.bin"; F1.bias_fn = "/b04.bin"; F1.act = ACT_RELU;

  FCNFile F2;
  F2.weight_fn = "/w05.bin"; F2.bias_fn = "/b05.bin"; F2.act = ACT_RELU;

  ConvMem D1c;
  D1c.K = 3; D1c.P = 65535; D1c.S = 2; D1c.OP = 0;
  D1c.weight = w06; D1c.bias = b06; D1c.act = ACT_RELU;

  ConvMem D2c;
  D2c.K = 3; D2c.P = 65535; D2c.S = 2; D2c.OP = 0;
  D2c.weight = w07; D2c.bias = b07; D2c.act = ACT_RELU;

  ConvMem D3c;
  D3c.K = 3; D3c.P = 65535; D3c.S = 2; D3c.OP = 0;
  D3c.weight = w08; D3c.bias = b08; D3c.act = ACT_RELU;

  ConvMem OUT;
  OUT.K = 3; OUT.P = 65535; OUT.S = 1; OUT.OP = 0;
  OUT.weight = w09; OUT.bias = b09; OUT.act = ACT_NONE;  // sigmoid applied manually.

  uint16_t W = W_IN;

  // Encoder stage 1:
  // [3][128][128] -> [32][128][128] -> maxpool -> [32][64][64]
  W = noodle_conv_float(X, C_IN, C1, Z1, W, E1, pool2, NULL);
#if AE_VERBOSE
  Serial.printf("enc1 W=%u\n", W);
  print_stats("Z1", Z1, (uint32_t)C1 * W * W);
#endif

  // Encoder stage 2:
  // [32][64][64] -> [64][64][64] -> maxpool -> [64][32][32]
  W = noodle_conv_float(Z1, C1, C2, Z2, W, E2, pool2, NULL);
#if AE_VERBOSE
  Serial.printf("enc2 W=%u\n", W);
  print_stats("Z2", Z2, (uint32_t)C2 * W * W);
#endif

  // Encoder stage 3:
  // [64][32][32] -> [128][32][32] -> maxpool -> [128][16][16]
  W = noodle_conv_float(Z2, C2, C3, Z3, W, E3, pool2, NULL);
#if AE_VERBOSE
  Serial.printf("enc3 W=%u\n", W);
  print_stats("Z3", Z3, (uint32_t)C3 * W * W);
#endif

  if (W != W3) return W;

  // Keras/TFLite flatten is NHWC. Noodle tensor is CHW, so use the existing
  // CHW -> HWC-like flatten helper.
  noodle_flat(Z3, FLAT, W3, C3);

  // Dense bottleneck: 32768 -> 64.
  noodle_fcn(FLAT, FLAT_N, LATENT, LAT, F1, NULL);

  // Dense expansion: 64 -> 32768.
  noodle_fcn(LAT, LATENT, FLAT_N, DENSE, F2, NULL);

  // Keras Reshape gives NHWC [16][16][128]. Convert to CHW for Noodle decoder.
  noodle_reshape(DENSE, R3, W3, C3);

  // Decoder:
  // [128][16][16] -> [128][32][32]
  W = noodle_conv_transpose_float(R3, C3, C4, D1, W3, D1c, NULL);
#if AE_VERBOSE
  Serial.printf("dec1 W=%u\n", W);
  print_stats("D1", D1, (uint32_t)C4 * W * W);
#endif

  // [128][32][32] -> [64][64][64]
  W = noodle_conv_transpose_float(D1, C4, C5, D2, W, D2c, NULL);
#if AE_VERBOSE
  Serial.printf("dec2 W=%u\n", W);
  print_stats("D2", D2, (uint32_t)C5 * W * W);
#endif

  // [64][64][64] -> [32][128][128]
  W = noodle_conv_transpose_float(D2, C5, C6, D3, W, D3c, NULL);
#if AE_VERBOSE
  Serial.printf("dec3 W=%u\n", W);
  print_stats("D3", D3, (uint32_t)C6 * W * W);
#endif

  // [32][128][128] -> [3][128][128]
  W = noodle_conv_transpose_float(D3, C6, C_OUT, Y, W, OUT, NULL);
#if AE_VERBOSE
  Serial.printf("out W=%u\n", W);
  print_stats("Y_pre_sigmoid", Y, (uint32_t)C_OUT * W * W);
#endif

  // Match final Keras activation="sigmoid".
  noodle_sigmoid(Y, (uint16_t)((uint32_t)C_OUT * W_OUT * W_OUT));

#if AE_VERBOSE
  print_stats("Y_sigmoid", Y, (uint32_t)C_OUT * W_OUT * W_OUT);
#endif

  return W;
}

static void process_one_image() {
  rgb888_hwc_to_chw_float_0_1(RX_BYTES, X, IMG_W);

  const uint32_t t0 = micros();
  const uint16_t W_final = run_autoencoder_forward();
  const uint32_t dt = micros() - t0;

  if (W_final != W_OUT) {
    Serial.printf("ERR_BAD_W %u\n", W_final);
    NoodleSerial::print_ready();
    return;
  }

  chw_float_0_1_to_rgb888_hwc(Y, TX_BYTES, IMG_W);

  NoodleSerial::send_output_image(TX_BYTES, IMG_SIZE, dt);
}

// ============================================================
// Arduino
// ============================================================

void setup() {
  NoodleSerial::begin();
  NoodleSerial::clear_input();

  Serial.println();
  Serial.println(F("BOOT OK Scoliosis RGB CAE"));

#if defined(ARDUINO_ARCH_ESP32)
setCpuFrequencyMhz(400); // set max CPU freq for best performance for P4!
  Serial.printf("CPU freq: %u MHz\n", getCpuFrequencyMhz());
  Serial.printf("Flash size: %u\n", ESP.getFlashChipSize());
  Serial.printf("PSRAM found: %s\n", psramFound() ? "YES" : "NO");
  Serial.printf("PSRAM size: %u\n", ESP.getPsramSize());
  Serial.printf("Free heap: %u\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %u\n", ESP.getFreePsram());
#endif

  while (!noodle_fs_init()) {
    delay(500);
    Serial.println(".");
  }
  Serial.println(F("FFAT OK!"));

  alloc_buffers();

  noodle_setup_temp_buffers((void *)TEMP2);

#if defined(ARDUINO_ARCH_ESP32)
  Serial.printf("Free heap after alloc: %u\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM after alloc: %u\n", ESP.getFreePsram());
#endif

  NoodleSerial::print_ready();
}

void loop() {
  // The board periodically re-announces READY while idle, so the Python sender
  // can recover even if it missed the first boot READY.
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
