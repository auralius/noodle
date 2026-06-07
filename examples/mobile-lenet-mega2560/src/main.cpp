#include <Arduino.h>
#include <avr/pgmspace.h>
#include "noodle.h"
#include "weights.h"
#include "biases.h"

// ============================================================
// Mobile-LeNet-16, Mega2560 PROGMEM-parameter + SD/RAM-activation version
// ============================================================
// Parameters:
//   w01 ... w14 and b01 ... b14 are stored in AVR PROGMEM.
//
// Activation flow:
//   Input 16x16 bytes from Serial
//   Stem + B1..B5: SD-backed activations
//   B5 output [24][4][4] copied to RAM
//   B6 + GAP + FCN: RAM activations
//   Activations remain SD/RAM-backed; only parameters move to PROGMEM.
// ============================================================

// ------------------------------------------------------------
// Board / serial / SD settings
// ------------------------------------------------------------
static const uint32_t BAUD = 115200;
static const uint32_t RX_TIMEOUT_MS = 3000;

#ifndef NOODLE_SD_CS
#define NOODLE_SD_CS 53   // Mega2560 SPI CS. Override from platformio.ini if needed.
#endif

// ------------------------------------------------------------
// Input settings
// ------------------------------------------------------------
static const uint16_t IMG_W = 16;
static const uint16_t IMG_H = 16;
static const uint16_t IMG_SIZE = IMG_W * IMG_H;

#define NORMALIZE_0_1

// ------------------------------------------------------------
// Temporary buffers
// ------------------------------------------------------------
// File-backed conv still needs two 16x16 temporary float planes.
static const uint16_t TEMP_FLOATS = IMG_W * IMG_H;

// Tail RAM after B5 is [24][4][4] = 384 floats.
static const uint16_t TAIL_C = 24;
static const uint16_t TAIL_W_MAX = 4;
static const uint16_t TAIL_FLOATS = TAIL_C * TAIL_W_MAX * TAIL_W_MAX;

// Final FCN: 24 -> 10. Keep activations/logits in SRAM, stream weights from SD.
static const uint16_t FCN_OUT = 10;

static float *TEMP1 = nullptr;
static float *TEMP2 = nullptr;
static float *ACT1  = nullptr;   // tail activation buffer
static float *ACT2  = nullptr;   // tail temporary buffer / FCN logits buffer
static uint8_t RX_BYTES[IMG_SIZE];

// ------------------------------------------------------------
// Activation files on SD
// ------------------------------------------------------------
static const char *F_IN  = "x00.bin";
static const char *F_A   = "a.bin";
static const char *F_TMP = "tmp.bin";

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms);
static bool image_bytes_to_file(const uint8_t *src, const char *fn, size_t n);

static void make_conv(ConvProgmem &c,
                      uint16_t K,
                      uint16_t P,
                      uint16_t S,
                      const float *w,
                      const float *b,
                      Activation act) {
  c.K = K;
  c.P = P;
  c.S = S;
  c.OP = 0;
  c.weight = w;
  c.bias = b;
  c.act = act;
}

// DW + PW block with SD-backed activations and PROGMEM-backed parameters.
// Input/output layout: packed CHW float tensors on SD.
static uint16_t dw_pw_block_sd_progmem(const char *in_fn,
                               const char *tmp_fn,
                               const char *out_fn,
                               uint16_t W_in,
                               uint16_t Cin,
                               uint16_t Cout,
                               uint16_t stride_dw,
                               const float *w_dw,
                               const float *b_dw,
                               const float *w_pw,
                               const float *b_pw) {
  Pool none;
  none.M = 1;
  none.T = 1;

  
  ConvProgmem dw;
  ConvProgmem pw;
  make_conv(dw, 3, 1, stride_dw, w_dw, b_dw, ACT_RELU);
  make_conv(pw, 1, 0, 1, w_pw, b_pw, ACT_RELU);
  
  uint16_t W = noodle_dwconv_float(in_fn, Cin, tmp_fn, W_in, dw, none, nullptr);
  W = noodle_conv_float(tmp_fn, Cin, Cout, out_fn, W, pw, none, nullptr);
  return W;
}

// DW + PW block, all RAM activations, PROGMEM-backed weights.
// Input/output layout: packed CHW float tensors in RAM.
// DW + PW block, RAM activations, PROGMEM-backed weights.
// Ping-pong style:
//   inout -> DW -> other
//   other -> PW -> inout
//
// Input/output layout: packed CHW float tensor in RAM.
// After return, the final block output is in `inout`.
static uint16_t dw_pw_block_mem_progmem_pingpong(float *inout,
                                         float *other,
                                         uint16_t W_in,
                                         uint16_t Cin,
                                         uint16_t Cout,
                                         uint16_t stride_dw,
                                         const float *w_dw,
                                         const float *b_dw,
                                         const float *w_pw,
                                         const float *b_pw) {
  Pool none;
  none.M = 1;
  none.T = 1;

  ConvProgmem dw;
  ConvProgmem pw;
  make_conv(dw, 3, 1, stride_dw, w_dw, b_dw, ACT_RELU);
  make_conv(pw, 1, 0, 1, w_pw, b_pw, ACT_RELU);

  uint16_t W = noodle_dwconv_float(inout, Cin, other, W_in, dw, none, nullptr);
  W = noodle_conv_float(other, Cin, Cout, inout, W, pw, none, nullptr);
  return W;
}

static void alloc_buffers() {
  TEMP1 = (float *)malloc((size_t)TEMP_FLOATS * sizeof(float));
  TEMP2 = (float *)malloc((size_t)TEMP_FLOATS * sizeof(float));
  ACT1  = (float *)malloc((size_t)TAIL_FLOATS * sizeof(float));
  ACT2  = (float *)malloc((size_t)TAIL_FLOATS * sizeof(float));
  if (!TEMP1 || !TEMP2 || !ACT1 || !ACT2) {
    Serial.println(F("ERROR malloc"));
    while (true) delay(1000);
  }

  noodle_setup_temp_buffers(TEMP1, TEMP2);
}


// ------------------------------------------------------------
// Inference
// ------------------------------------------------------------
static void predict() {
  const unsigned long t0 = micros();

  Pool none;
  none.M = 1;
  none.T = 1;

  // Input bytes -> SD float tensor: [1][16][16]
  if (!image_bytes_to_file(RX_BYTES, F_IN, IMG_SIZE)) {
    Serial.println(F("ERR input_file"));
    return;
  }

  // ---- Stem: file -> file, Conv3x3 1->8 ----
  ConvProgmem stem;
  make_conv(stem, 3, 1, 1, w01, b01, ACT_RELU);
  uint16_t W = noodle_conv_float(F_IN, 1, 8, F_A, IMG_W, stem, none, nullptr);
  if (W == 0) { Serial.println(F("ERR stem")); return; }

  // ---- B1: 8->8, stride 1 ----
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 8, 8, 1,
                     w02, b02, w03, b03);
  if (W == 0) { Serial.println(F("ERR B1")); return; }

  // ---- B2: 8->8, stride 1 ----
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 8, 8, 1,
                     w04, b04, w05, b05);
  if (W == 0) { Serial.println(F("ERR B2")); return; }

  // ---- B3: 8->16, stride 2 ----
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 8, 16, 2,
                     w06, b06, w07, b07);
  if (W == 0) { Serial.println(F("ERR B3")); return; }

  // ---- B4: 16->16, stride 1 ----
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 16, 16, 1,
                     w08, b08, w09, b09);
  if (W == 0) { Serial.println(F("ERR B4")); return; }

  // ---- B5: 16->24, stride 2 ----
  // Output should be [24][4][4] on SD.
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 16, 24, 2,
                     w10, b10, w11, b11);
  if (W == 0) { Serial.println(F("ERR B5")); return; }

  const uint16_t tail_n = (uint16_t)(24U * W * W);
  if (tail_n > TAIL_FLOATS) {
    Serial.println(F("ERR tail_size"));
    return;
  }

  // ---- Move tail activation SD -> RAM ----
  // F_A contains [24][W][W] after B5.
  noodle_array_from_file(F_A, ACT1, tail_n);

  // ---- B6: 24->24, stride 1, RAM -> RAM ----
  // ACT1 is input and output. ACT2 is the intermediate DW output.
  W = dw_pw_block_mem_progmem_pingpong(ACT1, ACT2, W, 24, 24, 1,
                             w12, b12, w13, b13);
  if (W == 0) { Serial.println(F("ERR B6_RAM")); return; }

  // ---- GAP: RAM [24][W][W] -> ACT1[0..23] in-place ----
  const uint16_t N = noodle_gap(ACT1, 24, W);
  if (N != 24) { Serial.println(F("ERR GAP_RAM")); return; }

  // ---- Dense: ACT1[24] -> ACT2[10], PROGMEM weights/bias ----
  // Ping-pong reuse:
  //   ACT1[0..23] = GAP vector input
  //   ACT2[0..9]  = logits / probabilities output
  uint16_t M = noodle_fcn_progmem(ACT1, N, FCN_OUT,
                                  ACT2,
                                  w14, b14,
                                  ACT_NONE,
                                  nullptr);
  if (M != FCN_OUT) { Serial.println(F("ERR FCN_PROGMEM")); return; }

  noodle_soft_max(ACT2, FCN_OUT);

  uint16_t pred = 0;
  float max_val = 0.0f;
  noodle_find_max(ACT2, FCN_OUT, max_val, pred);

  const float et = (float)(micros() - t0) * 1e-6f;

  Serial.print(F("PRED "));
  Serial.print(pred);
  Serial.print(' ');
  Serial.print(et, 4);
  Serial.print(' ');
  Serial.println(max_val, 4);
}

// ------------------------------------------------------------
// Arduino
// ------------------------------------------------------------
void setup() {
  Serial.begin(BAUD);
  delay(200);
  while (Serial.available()) Serial.read();

  // Mega2560 hardware SS must remain output for SPI master mode.
  pinMode(53, OUTPUT);
  digitalWrite(53, HIGH);

  if (!noodle_fs_init(NOODLE_SD_CS)) {
    Serial.println(F("ERR SD"));
    while (true) delay(1000);
  }

  alloc_buffers();

  Serial.println(F("READY"));
}

void loop() {
  if (!recv_exact(RX_BYTES, IMG_SIZE, RX_TIMEOUT_MS)) {
    Serial.println(F("READY"));
    return;
  }

  predict();
  Serial.println(F("READY"));
}

// ------------------------------------------------------------
// Helper implementations
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

static bool image_bytes_to_file(const uint8_t *src, const char *fn, size_t n) {
  NDL_File f = noodle_fs_open_write(fn);
  if (!f) return false;

#ifdef NORMALIZE_0_1
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) {
    noodle_write_float(f, (float)src[i] * inv);
  }
#else
  for (size_t i = 0; i < n; i++) {
    noodle_write_float(f, (float)src[i]);
  }
#endif

  f.close();
  return true;
}
