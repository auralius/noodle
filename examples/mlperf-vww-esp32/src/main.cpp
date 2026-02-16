#include <Arduino.h>
#include "noodle.h"
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
#include "w21.h"
#include "w22.h"
#include "w23.h"
#include "w24.h"
#include "w25.h"
#include "w26.h"
#include "w27.h"
#include "w28.h"

#include "b01.h"
#include "b02.h"
#include "b03.h"
#include "b04.h"
#include "b05.h"
#include "b06.h"
#include "b07.h"
#include "b08.h"
#include "b09.h"
#include "b10.h"
#include "b11.h"
#include "b12.h"
#include "b13.h"
#include "b14.h"
#include "b15.h"
#include "b16.h"
#include "b17.h"
#include "b18.h"
#include "b19.h"
#include "b20.h"
#include "b21.h"
#include "b22.h"
#include "b23.h"
#include "b24.h"
#include "b25.h"
#include "b26.h"
#include "b27.h"
#include "b28.h"

#define AUTO 65535

// =====================================
// Model constants
// =====================================
static constexpr uint16_t IN_W = 96;
static constexpr uint16_t IN_C = 3;
static constexpr uint32_t IN_PIX = (uint32_t)IN_W * (uint32_t)IN_W;
static constexpr uint32_t IN_RGB_BYTES = IN_PIX * 3;

// =====================================
// Buffers
// =====================================
static float *A   = nullptr;   // ping
static float *B   = nullptr;   // pong
static float *TMP = nullptr;   // scratch W*W (max 96*96)
static float *IN  = nullptr;   // 96*96*3 planar input (CHW)

static uint8_t *RGB = nullptr; // 96*96*3 interleaved bytes from serial

static const uint16_t RX_TIMEOUT_MS = 10000;

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
static void drain_serial(unsigned long ms = 50) {
  unsigned long t0 = millis();
  while (millis() - t0 < ms) {
    while (Serial.available()) Serial.read();
    delay(1);
  }
}

static bool recv_exact(uint8_t *dst, size_t n, unsigned long timeout_ms)
{
  unsigned long t0 = millis();
  size_t got = 0;

  while (got < n) {
    if ((millis() - t0) > timeout_ms) {
      Serial.printf("RX timeout: got=%u need=%u\n", (unsigned)got, (unsigned)n);
      return false;
    }
    int avail = Serial.available();
    if (avail <= 0) { delay(1); continue; }
    int r = Serial.readBytes((char *)(dst + got), n - got);
    if (r > 0) got += (size_t)r;
  }
  return true;
}

void bytes_to_float_image(const uint8_t *src, float *dst, size_t n)
{
#ifdef NORMALIZE_0_1
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i] * inv;
#else
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i];
#endif
}


// =====================================
// Minimal serial helpers
// =====================================
bool serial_read_exact(uint8_t *dst, uint32_t n) {
  uint32_t got = 0;
  while (got < n) {
    int r = Serial.readBytes((char*)dst + got, (size_t)(n - got));
    if (r > 0) got += (uint32_t)r;
    else delay(1);
  }
  return true;
}

uint32_t read_u32_le_blocking() {
  uint8_t b[4];
  serial_read_exact(b, 4);
  return (uint32_t)b[0] |
         ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
}

void rgb_u8_to_planar_float_0_1(const uint8_t *rgb, float *out_chw, uint16_t W) {
  const uint32_t n_pix = (uint32_t)W * (uint32_t)W;
  const float inv255 = 1.0f / 255.0f;
  for (uint32_t i = 0; i < n_pix; i++) {
    uint8_t r = rgb[3*i + 0];
    uint8_t g = rgb[3*i + 1];
    uint8_t b = rgb[3*i + 2];
    out_chw[i + 0*n_pix] = (float)r * inv255;
    out_chw[i + 1*n_pix] = (float)g * inv255;
    out_chw[i + 2*n_pix] = (float)b * inv255;
  }
}

// =====================================
// Allocate buffers
// =====================================
void alloc_buffers() {
  const uint32_t maxTensor = (uint32_t)48 * 48 * 32; // 36864 floats
  A   = (float*)malloc(maxTensor * sizeof(float));
  B   = (float*)malloc(maxTensor * sizeof(float));
  //TMP = (float*)malloc((uint32_t)IN_W * (uint32_t)IN_W * sizeof(float));      // 96*96 floats
  //IN  = (float*)malloc((uint32_t)IN_W * (uint32_t)IN_W * IN_C * sizeof(float)); // 96*96*3 floats
  RGB = (uint8_t*)malloc(IN_RGB_BYTES); // 27648 bytes
}

// =====================================
// Inference
// =====================================
void run_vww_on_IN_and_report() {
  Pool POOL_ID; POOL_ID.M = 1; POOL_ID.T = 1;

  uint16_t W = IN_W;
  uint16_t V = 0;

  ConvMem c00; c00.K=3; c00.P=AUTO; c00.S=2; c00.weight=w01; c00.bias=b01; c00.act=ACT_RELU;
  V = noodle_conv_float(B, 3, 8, A, W, c00, POOL_ID, nullptr); W = V;
  //Serial.println(V);

  ConvMem d01; d01.K=3; d01.P=1; d01.S=1; d01.weight=w02; d01.bias=b02; d01.act=ACT_RELU;
  V = noodle_dwconv_float(A, 8, B, W, d01, POOL_ID, nullptr); W = V;
  //Serial.println(V);

  ConvMem c02; c02.K=1; c02.P=0; c02.S=1; c02.weight=w03; c02.bias=b03; c02.act=ACT_RELU;
  V = noodle_conv_float(B, 8, 16, A, W, c02, POOL_ID, nullptr); W = V;
  //Serial.println(V);

  ConvMem d03; d03.K=3; d03.P=AUTO; d03.S=2; d03.weight=w04; d03.bias=b04; d03.act=ACT_RELU;
  V = noodle_dwconv_float(A, 16, B, W, d03, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c04; c04.K=1; c04.P=0; c04.S=1; c04.weight=w05; c04.bias=b05; c04.act=ACT_RELU;
  V = noodle_conv_float(B, 16, 32, A, W, c04, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d05; d05.K=3; d05.P=1; d05.S=1; d05.weight=w06; d05.bias=b06; d05.act=ACT_RELU;
  V = noodle_dwconv_float(A, 32, B, W, d05, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c06; c06.K=1; c06.P=0; c06.S=1; c06.weight=w07; c06.bias=b07; c06.act=ACT_RELU;
  V = noodle_conv_float(B, 32, 32, A, W, c06, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d07; d07.K=3; d07.P=AUTO; d07.S=2; d07.weight=w08; d07.bias=b08; d07.act=ACT_RELU;
  V = noodle_dwconv_float(A, 32, B, W, d07, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c08; c08.K=1; c08.P=0; c08.S=1; c08.weight=w09; c08.bias=b09; c08.act=ACT_RELU;
  V = noodle_conv_float(B, 32, 64, A, W, c08, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d09; d09.K=3; d09.P=1; d09.S=1; d09.weight=w10; d09.bias=b10; d09.act=ACT_RELU;
  V = noodle_dwconv_float(A, 64, B, W, d09, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c10; c10.K=1; c10.P=0; c10.S=1; c10.weight=w11; c10.bias=b11; c10.act=ACT_RELU;
  V = noodle_conv_float(B, 64, 64, A, W, c10, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d11; d11.K=3; d11.P=AUTO; d11.S=2; d11.weight=w12; d11.bias=b12; d11.act=ACT_RELU;
  V = noodle_dwconv_float(A, 64, B, W, d11, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c12; c12.K=1; c12.P=0; c12.S=1; c12.weight=w13; c12.bias=b13; c12.act=ACT_RELU;
  V = noodle_conv_float(B, 64, 128, A, W, c12, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d13; d13.K=3; d13.P=1; d13.S=1; d13.weight=w14; d13.bias=b14; d13.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d13, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c14; c14.K=1; c14.P=0; c14.S=1; c14.weight=w15; c14.bias=b15; c14.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c14, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d15; d15.K=3; d15.P=1; d15.S=1; d15.weight=w16; d15.bias=b16; d15.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d15, POOL_ID, nullptr); W = V;
  //Serial.println(V);

  ConvMem c16; c16.K=1; c16.P=0; c16.S=1; c16.weight=w17; c16.bias=b17; c16.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c16, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d17; d17.K=3; d17.P=1; d17.S=1; d17.weight=w18; d17.bias=b18; d17.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d17, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c18; c18.K=1; c18.P=0; c18.S=1; c18.weight=w19; c18.bias=b19; c18.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c18, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d19; d19.K=3; d19.P=1; d19.S=1; d19.weight=w20; d19.bias=b20; d19.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d19, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c20; c20.K=1; c20.P=0; c20.S=1; c20.weight=w21; c20.bias=b21; c20.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c20, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d21; d21.K=3; d21.P=1; d21.S=1; d21.weight=w22; d21.bias=b22; d21.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d21, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c22; c22.K=1; c22.P=0; c22.S=1; c22.weight=w23; c22.bias=b23; c22.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c22, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d23; d23.K=3; d23.P=AUTO; d23.S=2; d23.weight=w24; d23.bias=b24; d23.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d23, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c24; c24.K=1; c24.P=0; c24.S=1; c24.weight=w25; c24.bias=b25; c24.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 256, A, W, c24, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem d25; d25.K=3; d25.P=1; d25.S=1; d25.weight=w26; d25.bias=b26; d25.act=ACT_RELU;
  V = noodle_dwconv_float(A, 256, B, W, d25, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  ConvMem c26; c26.K=1; c26.P=0; c26.S=1; c26.weight=w27; c26.bias=b27; c26.act=ACT_RELU;
  V = noodle_conv_float(B, 256, 256, A, W, c26, POOL_ID, nullptr); W = V;
  //Serial.println(V);
  
  uint16_t C = noodle_gap(A, 256, W);
  //Serial.println(C);
  
  float out2[2];
  FCNMem fcf; fcf.weight = w28; fcf.bias = b28; fcf.act = ACT_SOFTMAX;
  (void)noodle_fcn((const float*)A, 256, 2, out2, fcf, nullptr);

  uint8_t pred = (out2[1] > out2[0]) ? 1 : 0;
  Serial.printf("ms=%lu P0=%.6f P1=%.6f pred=%u\n", (unsigned long)0, out2[0], out2[1], pred);
}

// =====================================
// Arduino setup/loop
// =====================================
void setup() {
  Serial.begin(115200);
  delay(200);

  //while (!noodle_fs_init()) {
  //  Serial.println("FS init failed, retry...");
  //  delay(500);
  //}
  //Serial.println("FS OK");

  alloc_buffers();
  if (!A || !B || !RGB) {
    Serial.println("ERR malloc");
    while (1) delay(100);
  }

  //noodle_setup_temp_buffers((void*)TMP, (void*)B);

  Serial.println("READY"); // Python can wait for this
}

void loop() {
  if (!recv_exact(RGB, IN_RGB_BYTES, RX_TIMEOUT_MS)) {
    drain_serial(50);
    Serial.println(F("READY"));
      return;
    }
  bytes_to_float_image(RGB, B, IN_RGB_BYTES);


  // ConvMemert to planar float [0,1]
  rgb_u8_to_planar_float_0_1(RGB, B, IN_W);

  // Run inference + report time
  unsigned long t0 = millis();
  run_vww_on_IN_and_report();
  unsigned long t1 = millis();

  // Overwrite ms=0 line by printing a second timing line (keeps code simple)
  // If you prefer single-line output, see note below.
  Serial.printf("time_ms=%lu\n", (unsigned long)(t1 - t0));
}