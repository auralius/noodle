// MobileLenet with serial RX of raw 28x28 grayscale image. 
// Last tested on ESP32, June 10, 2026
// Auralius Manurung

#include <Arduino.h>
#include "noodle.h"

// ------------------------------------------------------------
// Exported parameters
// ------------------------------------------------------------
#include "w01.h"   // stem conv3x3 (1->8)
#include "w02.h"   // B1 DW3x3 (8ch)
#include "w03.h"   // B1 PW1x1 (8->8)
#include "w04.h"   // B2 DW3x3 (8ch)
#include "w05.h"   // B2 PW1x1 (8->8)
#include "w06.h"   // B3 DW3x3 s2 (8ch)
#include "w07.h"   // B3 PW1x1 (8->16)
#include "w08.h"   // B4 DW3x3 (16ch)
#include "w09.h"   // B4 PW1x1 (16->16)
#include "w10.h"   // B5 DW3x3 s2 (16ch)
#include "w11.h"   // B5 PW1x1 (16->24)
#include "w12.h"   // B6 DW3x3 (24ch)
#include "w13.h"   // B6 PW1x1 (24->24)
#include "w14.h"   // Dense (24->10)

#include "b01.h"   // stem conv bias (8)
#include "b02.h"   // Dense bias (10)

// BatchNorm packed as: [gamma(C), beta(C), mean(C), var(C)]
#include "bn01.h"
#include "bn02.h"
#include "bn03.h"
#include "bn04.h"
#include "bn05.h"
#include "bn06.h"
#include "bn07.h"
#include "bn08.h"
#include "bn09.h"
#include "bn10.h"
#include "bn11.h"
#include "bn12.h"
#include "bn13.h"

// ------------------------------------------------------------
// Serial RX protocol settings
// ------------------------------------------------------------
static const uint32_t BAUD = 115200;
static const uint32_t RX_TIMEOUT_MS = 3000;

static const uint16_t IMG_W = 28;
static const uint16_t IMG_H = 28;
static const uint16_t IMG_SIZE = IMG_W * IMG_H;

#define NORMALIZE_0_1

// ------------------------------------------------------------
// Smart grow-only buffers
// ------------------------------------------------------------
static NoodleBuffer FEAT_A;
static NoodleBuffer FEAT_B;

static uint8_t RX_BYTES[IMG_SIZE];

static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms);
static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n);

// ------------------------------------------------------------
// Depthwise + Pointwise block
//
// DW: in  -> out  (+ BN + ReLU on out)
// PW: out -> in   (+ BN + ReLU on in)
//
// Final output lives in `in` again.
// ------------------------------------------------------------
static uint16_t noodle_dw_pw_block(NoodleBuffer *in,
                                   NoodleBuffer *out,
                                   uint16_t W_in,
                                   uint16_t Cin,
                                   uint16_t Cout,
                                   uint16_t stride_dw,
                                   const float *w_dw,
                                   const float *bn_dw,
                                   const float *w_pw,
                                   const float *bn_pw) {
  const float BN_EPS = 1e-3f;

  Pool none{};
  none.M = 1;
  none.T = 1;

  ConvMem dw{};
  dw.K = 3;
  dw.P = 1;
  dw.S = stride_dw;
  dw.weight = w_dw;
  dw.bias = nullptr;
  dw.act = ACT_NONE;

  uint16_t W = noodle_dwconv_float(in, Cin, out, W_in, dw, none, nullptr);
  noodle_bn_relu(out->data, Cin, W, bn_dw, BN_EPS);

  ConvMem pw{};
  pw.K = 1;
  pw.P = 0;
  pw.S = 1;
  pw.weight = w_pw;
  pw.bias = nullptr;
  pw.act = ACT_NONE;

  W = noodle_conv_float(out, Cin, Cout, in, W, pw, none, nullptr);
  noodle_bn_relu(in->data, Cout, W, bn_pw, BN_EPS);

  return W;
}

// ------------------------------------------------------------
// Inference
// ------------------------------------------------------------
static void predict() {
  const unsigned long t0 = micros();

  Pool none{};
  none.M = 1;
  none.T = 1;

  ConvMem stem{};
  stem.K = 3;
  stem.P = 1;
  stem.S = 1;
  stem.weight = w01;
  stem.bias = b01;
  stem.act = ACT_NONE;

  FCNMem head{};
  head.weight = w14;
  head.bias = b02;
  head.act = ACT_NONE;

  NoodleBuffer *in = &FEAT_A;
  NoodleBuffer *out = &FEAT_B;

  // Stem: Conv3x3 (1->8) + BN + ReLU
  uint16_t W = noodle_conv_float(in, 1, 8, out, IMG_W, stem, none, nullptr);
  noodle_bn_relu(out->data, 8, W, bn01, 1e-3f);

  // After stem, feature map is in FEAT_B.
  in = &FEAT_B;
  out = &FEAT_A;

  W = noodle_dw_pw_block(in, out, W, 8,  8,  1, w02, bn02, w03, bn03);
  W = noodle_dw_pw_block(in, out, W, 8,  8,  1, w04, bn04, w05, bn05);
  W = noodle_dw_pw_block(in, out, W, 8,  16, 2, w06, bn06, w07, bn07);
  W = noodle_dw_pw_block(in, out, W, 16, 16, 1, w08, bn08, w09, bn09);
  W = noodle_dw_pw_block(in, out, W, 16, 24, 2, w10, bn10, w11, bn11);
  W = noodle_dw_pw_block(in, out, W, 24, 24, 1, w12, bn12, w13, bn13);

  // GAP: (W x W x 24) -> (24,), in-place in `in`.
  uint16_t N = noodle_gap(in->data, 24, W);

  // Dense: 24 -> 10. Logits are written to `out`.
  N = noodle_fcn(in, N, 10, out, head, nullptr);

  noodle_soft_max(out, N);

  uint16_t pred;
  float max_val;
  noodle_find_max(out, N, max_val, pred);

  const float et = (float)(micros() - t0) * 1e-6f;

  Serial.print(F("PRED "));
  Serial.print(pred);
  Serial.print(' ');
  Serial.print(et, 4);
  Serial.print(' ');
  Serial.print(max_val, 4);
  Serial.println();
}

// ------------------------------------------------------------
// Arduino
// ------------------------------------------------------------
void setup() {
  Serial.begin(BAUD);
  delay(200);
  while (Serial.available()) Serial.read();

  noodle_buffer_init(&FEAT_A);
  noodle_buffer_init(&FEAT_B);

  Serial.println(F("READY"));
  delay(1000);

  float *x = noodle_buffer_require(&FEAT_A, IMG_SIZE);
  bytes_to_float_image(RX_BYTES, x, IMG_SIZE);
  predict();
}

void loop() {
  if (!recv_exact(RX_BYTES, IMG_SIZE, RX_TIMEOUT_MS)) {
    Serial.println(F("READY"));
    return;
  }

  float *x = noodle_buffer_require(&FEAT_A, IMG_SIZE);
  bytes_to_float_image(RX_BYTES, x, IMG_SIZE);
  predict();
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms) {
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

static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n) {
#ifdef NORMALIZE_0_1
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i] * inv;
#else
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i];
#endif
}
