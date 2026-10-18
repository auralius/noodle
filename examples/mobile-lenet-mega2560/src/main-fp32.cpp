#include <Arduino.h>
#include <avr/pgmspace.h>
#include "noodle.h"
#include "fp32/weights.h"
#include "fp32/biases.h"

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
#ifndef NOODLE_SD_SCK_MHZ
#define NOODLE_SD_SCK_MHZ 4
#endif
// Independent of inference arithmetic; optional diagnostic timestamp collection.
#ifndef CASE2_FP32_STAGE_TRACE
#define CASE2_FP32_STAGE_TRACE 1
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

// Final FCN: 24 -> 10. Keep activations/logits in SRAM and weights in PROGMEM.
static const uint16_t FCN_OUT = 10;

static float *TEMP1 = nullptr;
static float *TEMP2 = nullptr;
// The current Noodle RAM-layer API requires NoodleBuffer, not raw float*.
// Registered buffers share the packed arena; reserve both maximum sizes in setup.
static NoodleBuffer ACT1 = {};  // tail activation buffer
static NoodleBuffer ACT2 = {};  // tail temporary buffer / FCN logits buffer
static uint8_t RX_BYTES[IMG_SIZE];
#if CASE2_FP32_STAGE_TRACE
static uint32_t stage_us[14];
#endif

#if defined(__AVR__)
extern char __heap_start;
extern char *__brkval;
static int free_ram() {
  char sp;
  return (int)(&sp - (__brkval ? (char*)__brkval : &__heap_start));
}
#else
static int free_ram() { return -1; }
#endif

static void print_stage_trace() {
#if CASE2_FP32_STAGE_TRACE
  for (uint8_t i = 0; i < 14; ++i) {
    Serial.print(F("STAGE "));
    Serial.print(i + 1);
    Serial.print(' ');
    Serial.println(stage_us[i]);
  }
#endif
}


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
                               const float *b_pw,
                               uint8_t dw_stage) {
  Pool none;
  none.M = 1;
  none.T = 1;

  
  ConvProgmem dw;
  ConvProgmem pw;
  make_conv(dw, 3, 1, stride_dw, w_dw, b_dw, ACT_RELU);
  make_conv(pw, 1, 0, 1, w_pw, b_pw, ACT_RELU);
  
#if CASE2_FP32_STAGE_TRACE
  const uint32_t tdw = micros();
#endif
  uint16_t W = noodle_dwconv_float(in_fn, Cin, tmp_fn, W_in, dw, none, nullptr);
#if CASE2_FP32_STAGE_TRACE
  stage_us[dw_stage - 1] = micros() - tdw;
  const uint32_t tpw = micros();
#endif
  W = noodle_conv_float(tmp_fn, Cin, Cout, out_fn, W, pw, none, nullptr);
#if CASE2_FP32_STAGE_TRACE
  stage_us[dw_stage] = micros() - tpw;
#endif
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
static uint16_t dw_pw_block_mem_progmem_pingpong(NoodleBuffer *inout,
                                         NoodleBuffer *other,
                                         uint16_t W_in,
                                         uint16_t Cin,
                                         uint16_t Cout,
                                         uint16_t stride_dw,
                                         const float *w_dw,
                                         const float *b_dw,
                                         const float *w_pw,
                                         const float *b_pw,
    uint8_t dw_stage) {
  Pool none;
  none.M = 1;
  none.T = 1;

  ConvProgmem dw;
  ConvProgmem pw;
  make_conv(dw, 3, 1, stride_dw, w_dw, b_dw, ACT_RELU);
  make_conv(pw, 1, 0, 1, w_pw, b_pw, ACT_RELU);

#if CASE2_FP32_STAGE_TRACE
  const uint32_t tdw = micros();
#endif
  uint16_t W = noodle_dwconv_float(inout, Cin, other, W_in, dw, none, nullptr);
#if CASE2_FP32_STAGE_TRACE
  stage_us[dw_stage - 1] = micros() - tdw;
  const uint32_t tpw = micros();
#endif
  W = noodle_conv_float(other, Cin, Cout, inout, W, pw, none, nullptr);
#if CASE2_FP32_STAGE_TRACE
  stage_us[dw_stage] = micros() - tpw;
#endif
  return W;
}

static void alloc_buffers() {
  // Allocate the packed B6 arena BEFORE legacy scratch buffers, reducing
  // fragmentation risk on the Mega's 8 KB SRAM.  This pair needs 3072 bytes.
  noodle_buffer_init(&ACT1);
  noodle_buffer_init(&ACT2);
  if (!noodle_buffer_require(&ACT1, TAIL_FLOATS) ||
      !noodle_buffer_require(&ACT2, TAIL_FLOATS)) {
    Serial.println(F("ERROR B6 buffer arena"));
    while (true) delay(1000);
  }
  TEMP1 = (float *)malloc((size_t)TEMP_FLOATS * sizeof(float));
  TEMP2 = (float *)malloc((size_t)TEMP_FLOATS * sizeof(float));
  if (!TEMP1 || !TEMP2) {
    Serial.println(F("ERROR scratch malloc"));
    while (true) delay(1000);
  }
  noodle_setup_temp_buffers(TEMP1, TEMP2);
}


// ------------------------------------------------------------
// Inference
// ------------------------------------------------------------
static void predict() {
  const unsigned long t0 = micros();
#if CASE2_FP32_STAGE_TRACE
  for (uint8_t i = 0; i < 14; ++i) stage_us[i] = 0;
#endif

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
#if CASE2_FP32_STAGE_TRACE
  const uint32_t tstem = micros();
#endif
  uint16_t W = noodle_conv_float(F_IN, 1, 8, F_A, IMG_W, stem, none, nullptr);
#if CASE2_FP32_STAGE_TRACE
  stage_us[0] = micros() - tstem;
#endif
  if (W == 0) { Serial.println(F("ERR stem")); return; }

  // ---- B1: 8->8, stride 1 ----
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 8, 8, 1,
                     w02, b02, w03, b03, 2);
  if (W == 0) { Serial.println(F("ERR B1")); return; }

  // ---- B2: 8->8, stride 1 ----
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 8, 8, 1,
                     w04, b04, w05, b05, 4);
  if (W == 0) { Serial.println(F("ERR B2")); return; }

  // ---- B3: 8->16, stride 2 ----
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 8, 16, 2,
                     w06, b06, w07, b07, 6);
  if (W == 0) { Serial.println(F("ERR B3")); return; }

  // ---- B4: 16->16, stride 1 ----
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 16, 16, 1,
                     w08, b08, w09, b09, 8);
  if (W == 0) { Serial.println(F("ERR B4")); return; }

  // ---- B5: 16->24, stride 2 ----
  // Output should be [24][4][4] on SD.
  W = dw_pw_block_sd_progmem(F_A, F_TMP, F_A, W, 16, 24, 2,
                     w10, b10, w11, b11, 10);
  if (W == 0) { Serial.println(F("ERR B5")); return; }

  const uint16_t tail_n = (uint16_t)(24U * W * W);
  if (tail_n > TAIL_FLOATS) {
    Serial.println(F("ERR tail_size"));
    return;
  }

  // ---- Move tail activation SD -> RAM ----
  // F_A contains [24][W][W] after B5.
  noodle_array_from_file(F_A, ACT1.data, tail_n);

  // ---- B6: 24->24, stride 1, RAM -> RAM ----
  // ACT1 is input and output. ACT2 is the intermediate DW output.
  W = dw_pw_block_mem_progmem_pingpong(&ACT1, &ACT2, W, 24, 24, 1,
                             w12, b12, w13, b13, 12);
  if (W == 0) { Serial.println(F("ERR B6_RAM")); return; }

  // ---- GAP: RAM [24][W][W] -> ACT1[0..23] in-place ----
  const uint16_t N = noodle_gap(&ACT1, 24, W);
  if (N != 24) { Serial.println(F("ERR GAP_RAM")); return; }

  // ---- Dense: ACT1[24] -> ACT2[10], PROGMEM weights/bias ----
  // Ping-pong reuse:
  //   ACT1[0..23] = GAP vector input
  //   ACT2[0..9]  = logits / probabilities output
#if CASE2_FP32_STAGE_TRACE
  const uint32_t tdense = micros();
#endif
  uint16_t M = noodle_fcn_progmem(&ACT1, N, FCN_OUT,
                                  &ACT2,
                                  w14, b14,
                                  ACT_NONE,
                                  nullptr);
#if CASE2_FP32_STAGE_TRACE
  stage_us[13] = micros() - tdense;
#endif
  if (M != FCN_OUT) { Serial.println(F("ERR FCN_PROGMEM")); return; }

  noodle_soft_max(&ACT2, FCN_OUT);

  uint16_t pred = 0;
  float max_val = 0.0f;
  noodle_find_max(&ACT2, FCN_OUT, max_val, pred);

  const float et = (float)(micros() - t0) * 1e-6f;

  Serial.print(F("PRED "));
  Serial.print(pred);
  Serial.print(' ');
  Serial.print(et, 4);
  Serial.print(' ');
  Serial.println(max_val, 4);
  Serial.print(F("MEM SNAP free_ram="));
  Serial.println(free_ram());
  print_stage_trace();
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
  Serial.print(F("DIAG FP32 SD_SPI_MHZ="));
  Serial.print(NOODLE_SD_SCK_MHZ);
  Serial.print(F(" STAGE_TRACE="));
  Serial.print(CASE2_FP32_STAGE_TRACE);
  Serial.print(F(" free_ram="));
  Serial.println(free_ram());
  Serial.println(F("READY"));
}

void loop() {
  // Match INT8 diagnostic firmware: no unsolicited READY messages while idle.
  if (!Serial.available()) return;
  if (!recv_exact(RX_BYTES, IMG_SIZE, RX_TIMEOUT_MS)) {
    Serial.println(F("ERR RX_PARTIAL"));
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
