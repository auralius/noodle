#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include "noodle.h"

// ------------------------------------------------------------
// Exported parameters
// ------------------------------------------------------------

// Weights
#include "pneumonia/w01.h"
#include "pneumonia/w02.h"
#include "pneumonia/w03.h"
#include "pneumonia/w04.h"
#include "pneumonia/w05.h"
#include "pneumonia/w06.h"
#include "pneumonia/w07.h"
#include "pneumonia/w08.h"
#include "pneumonia/w09.h"
#include "pneumonia/w10.h"
#include "pneumonia/w11.h"
#include "pneumonia/w12.h"
#include "pneumonia/w13.h"
#include "pneumonia/w14.h"

// Biases
#include "pneumonia/b01.h"
#include "pneumonia/b02.h"
#include "pneumonia/b03.h"
#include "pneumonia/b04.h"
#include "pneumonia/b05.h"
#include "pneumonia/b06.h"
#include "pneumonia/b07.h"
#include "pneumonia/b08.h"
#include "pneumonia/b09.h"
#include "pneumonia/b10.h"
#include "pneumonia/b11.h"
#include "pneumonia/b12.h"
#include "pneumonia/b13.h"
#include "pneumonia/b14.h"

// ------------------------------------------------------------
// Serial RX protocol settings
// ------------------------------------------------------------
static const uint32_t BAUD = 921600;
static const uint32_t RX_TIMEOUT_MS = 5000;

static const uint16_t IMG_W = 64;
static const uint16_t IMG_H = 64;
static const uint32_t IMG_SIZE = (uint32_t)IMG_W * IMG_H;

static const uint8_t FRAME_MAGIC[4] = {'I', 'M', 'G', '0'};

#define NORMALIZE_0_1

// ------------------------------------------------------------
// Buffers
// Peak activation:
//   64x64x6  = 24576
//   32x32x24 = 24576
// ------------------------------------------------------------
static const uint32_t MAX_FEAT_FLOATS = (uint32_t)64 * 64 * 6;

static float *FEAT_A = nullptr;   // ping
static float *FEAT_B = nullptr;   // pong
static uint8_t RX_BYTES[IMG_SIZE];

// ------------------------------------------------------------
// Forward decl
// ------------------------------------------------------------
static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms);
static bool recv_frame_image(uint8_t *dst, size_t expected_n, uint32_t timeout_ms);
static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n);

static inline uint32_t tensor_bytes(uint16_t W, uint16_t C)
{
  return (uint32_t)W * (uint32_t)W * (uint32_t)C * (uint32_t)sizeof(float);
}

static void send_layer_stat(const char *name,
                            uint32_t time_us,
                            uint16_t Win, uint16_t Cin,
                            uint16_t Wout, uint16_t Cout)
{
  const uint32_t in_bytes   = tensor_bytes(Win, Cin);
  const uint32_t out_bytes  = tensor_bytes(Wout, Cout);
  const uint32_t live_bytes = in_bytes + out_bytes;

  Serial.print(F("LAYER "));
  Serial.print(name);
  Serial.print(' ');
  Serial.print(time_us);
  Serial.print(' ');
  Serial.print(in_bytes);
  Serial.print(' ');
  Serial.print(out_bytes);
  Serial.print(' ');
  Serial.println(live_bytes);
}

static void send_mem_buffers_once()
{
  const uint32_t feat_a_bytes = MAX_FEAT_FLOATS * sizeof(float);
  const uint32_t feat_b_bytes = MAX_FEAT_FLOATS * sizeof(float);
  const uint32_t rx_bytes     = sizeof(RX_BYTES);
  const uint32_t total_bytes  = feat_a_bytes + feat_b_bytes + rx_bytes;

  Serial.print(F("MEMBUF FEAT_A "));
  Serial.println(feat_a_bytes);

  Serial.print(F("MEMBUF FEAT_B "));
  Serial.println(feat_b_bytes);

  Serial.print(F("MEMBUF RX_BYTES "));
  Serial.println(rx_bytes);

  Serial.print(F("MEMBUF TOTAL_PERSISTENT "));
  Serial.println(total_bytes);
}

// ------------------------------------------------------------
// Depthwise + Pointwise block
//   DW: Cin -> Cin
//   PW: Cin -> Cout
// Final output lives again in 'in'.
// ------------------------------------------------------------
static uint16_t noodle_dw_pw_block_folded(float *in,
                                          float *out,
                                          uint16_t W_in,
                                          uint16_t Cin,
                                          uint16_t Cout,
                                          uint16_t stride_dw,
                                          const float *w_dw,
                                          const float *b_dw,
                                          const float *w_pw,
                                          const float *b_pw,
                                          const char *name_dw,
                                          const char *name_pw)
{
  Pool none{};
  none.M = 1;
  none.T = 1;

  ConvMem dw{};
  dw.K = 3;
  dw.P = 1;
  dw.S = stride_dw;
  dw.weight = w_dw;
  dw.bias   = b_dw;
  dw.act    = ACT_RELU;

  const unsigned long t_dw0 = micros();
  uint16_t W_mid = noodle_dwconv_float(in, Cin, out, W_in, dw, none, nullptr);
  const uint32_t t_dw_us = (uint32_t)(micros() - t_dw0);

  send_layer_stat(name_dw, t_dw_us, W_in, Cin, W_mid, Cin);

  ConvMem pw{};
  pw.K = 1;
  pw.P = 0;
  pw.S = 1;
  pw.weight = w_pw;
  pw.bias   = b_pw;
  pw.act    = ACT_RELU;

  const unsigned long t_pw0 = micros();
  uint16_t W_out = noodle_conv_float(out, Cin, Cout, in, W_mid, pw, none, nullptr);
  const uint32_t t_pw_us = (uint32_t)(micros() - t_pw0);

  send_layer_stat(name_pw, t_pw_us, W_mid, Cin, W_out, Cout);

  return W_out;
}

static void alloc_buffers()
{
  FEAT_A = (float *)malloc(MAX_FEAT_FLOATS * sizeof(float));
  FEAT_B = (float *)malloc(MAX_FEAT_FLOATS * sizeof(float));

  if (!FEAT_A || !FEAT_B) {
    Serial.println(F("ERROR: malloc failed (out of RAM)"));
    while (true) delay(1000);
  }
}

// ------------------------------------------------------------
// Inference
// ------------------------------------------------------------
void predict()
{
  const unsigned long t0 = micros();

  Pool none{};
  none.M = 1;
  none.T = 1;

  ConvMem stem{};
  stem.K = 3;
  stem.P = 1;
  stem.S = 1;
  stem.weight = w01;
  stem.bias   = b01;
  stem.act    = ACT_RELU;

  FCNMem head{};
  head.weight = w14;
  head.bias   = b14;
  head.act    = ACT_NONE;

  const unsigned long t_stem0 = micros();
  uint16_t W = noodle_conv_float(FEAT_A, 1, 6, FEAT_B, IMG_W, stem, none, nullptr);
  const uint32_t t_stem_us = (uint32_t)(micros() - t_stem0);
  send_layer_stat("stem", t_stem_us, 64, 1, W, 6);

  float *in  = FEAT_B;
  float *out = FEAT_A;

  W = noodle_dw_pw_block_folded(in, out, W, 6, 6, 1,
                                w02, b02, w03, b03,
                                "b1_dw", "b1_pw");

  W = noodle_dw_pw_block_folded(in, out, W, 6, 6, 1,
                                w04, b04, w05, b05,
                                "b2_dw", "b2_pw");

  W = noodle_dw_pw_block_folded(in, out, W, 6, 24, 2,
                                w06, b06, w07, b07,
                                "b3_dw", "b3_pw");

  W = noodle_dw_pw_block_folded(in, out, W, 24, 24, 1,
                                w08, b08, w09, b09,
                                "b4_dw", "b4_pw");

  W = noodle_dw_pw_block_folded(in, out, W, 24, 24, 2,
                                w10, b10, w11, b11,
                                "b5_dw", "b5_pw");

  W = noodle_dw_pw_block_folded(in, out, W, 24, 12, 1,
                                w12, b12, w13, b13,
                                "b6_dw", "b6_pw");

  {
    const uint16_t W_in = W;
    const unsigned long t_gap0 = micros();
    W = noodle_gap(in, 12, W);
    const uint32_t t_gap_us = (uint32_t)(micros() - t_gap0);

    const uint32_t in_bytes   = tensor_bytes(W_in, 12);
    const uint32_t out_bytes  = (uint32_t)12 * sizeof(float);
    const uint32_t live_bytes = in_bytes;

    Serial.print(F("LAYER gap "));
    Serial.print(t_gap_us);
    Serial.print(' ');
    Serial.print(in_bytes);
    Serial.print(' ');
    Serial.print(out_bytes);
    Serial.print(' ');
    Serial.println(live_bytes);
  }

  {
    const unsigned long t_fcn0 = micros();
    W = noodle_fcn(in, W, 1, out, head, nullptr);
    const uint32_t t_fcn_us = (uint32_t)(micros() - t_fcn0);

    const uint32_t in_bytes   = (uint32_t)12 * sizeof(float);
    const uint32_t out_bytes  = (uint32_t)1 * sizeof(float);
    const uint32_t live_bytes = in_bytes + out_bytes;

    Serial.print(F("LAYER dense "));
    Serial.print(t_fcn_us);
    Serial.print(' ');
    Serial.print(in_bytes);
    Serial.print(' ');
    Serial.print(out_bytes);
    Serial.print(' ');
    Serial.println(live_bytes);
  }

  {
    const unsigned long t_sig0 = micros();
    out[0] = noodle_sigmoidf(out[0]);
    const uint32_t t_sig_us = (uint32_t)(micros() - t_sig0);

    const uint32_t bytes = (uint32_t)1 * sizeof(float);

    Serial.print(F("LAYER sigmoid "));
    Serial.print(t_sig_us);
    Serial.print(' ');
    Serial.print(bytes);
    Serial.print(' ');
    Serial.print(bytes);
    Serial.print(' ');
    Serial.println(bytes);
  }

  const float et = (float)(micros() - t0) * 1e-6f;
  const float p_pneumonia = out[0];
  const uint16_t pred = (p_pneumonia >= 0.5f) ? 1 : 0;

  Serial.print(F("PRED "));
  Serial.print(pred);
  Serial.print(' ');
  Serial.print(et, 4);
  Serial.print(' ');
  Serial.print(p_pneumonia, 6);
  Serial.println();

  Serial.println(F("READY"));
}

// ------------------------------------------------------------
// Arduino
// ------------------------------------------------------------
void setup()
{
  Serial.begin(BAUD);
  Serial.setTimeout(20);
  delay(400);
  while (Serial.available()) Serial.read();

  alloc_buffers();
  send_mem_buffers_once();

  Serial.println(F("READY"));
}

void loop()
{
  if (!recv_frame_image(RX_BYTES, IMG_SIZE, RX_TIMEOUT_MS)) {
    delay(2);
    return;
  }

  Serial.println(F("GOT_FRAME"));
  bytes_to_float_image(RX_BYTES, FEAT_A, IMG_SIZE);
  predict();
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms)
{
  uint32_t last_rx = millis();
  size_t got = 0;

  while (got < n) {
    int avail = Serial.available();

    if (avail > 0) {
      size_t want = (size_t)avail;
      if (want > (n - got)) want = n - got;

      int r = Serial.readBytes((char *)(dst + got), want);
      if (r > 0) {
        got += (size_t)r;
        last_rx = millis();
      }
    } else {
      if ((millis() - last_rx) > timeout_ms) {
        return false;
      }
      delay(1);
    }
  }
  return true;
}

static bool recv_frame_image(uint8_t *dst, size_t expected_n, uint32_t timeout_ms)
{
  static uint8_t hdr[8];

  while (Serial.available() > 0) {
    int b = Serial.read();
    if (b < 0) break;

    if ((uint8_t)b != FRAME_MAGIC[0]) {
      continue;
    }

    if (!recv_exact(&hdr[1], 3, timeout_ms)) {
      return false;
    }
    hdr[0] = FRAME_MAGIC[0];

    if (hdr[1] != FRAME_MAGIC[1] || hdr[2] != FRAME_MAGIC[2] || hdr[3] != FRAME_MAGIC[3]) {
      continue;
    }

    if (!recv_exact(&hdr[4], 4, timeout_ms)) {
      return false;
    }

    uint32_t n = (uint32_t)hdr[4]
               | ((uint32_t)hdr[5] << 8)
               | ((uint32_t)hdr[6] << 16)
               | ((uint32_t)hdr[7] << 24);

    if (n != expected_n) {
      Serial.print(F("BAD_LEN "));
      Serial.println((unsigned long)n);
      return false;
    }

    return recv_exact(dst, expected_n, timeout_ms);
  }

  return false;
}

static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n)
{
#ifdef NORMALIZE_0_1
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i] * inv;
#else
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i];
#endif
}

