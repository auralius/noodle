#include <Arduino.h>
#include "noodle.h"

// ===============================
// USER: fill these paths exactly
// ===============================
static const char *W_OP00 = "/w01.txt";  static const char *B_OP00 = "/b01.txt";
static const char *W_OP01 = "/w02.txt";  static const char *B_OP01 = "/b02.txt";
static const char *W_OP02 = "/w03.txt";  static const char *B_OP02 = "/b03.txt";
static const char *W_OP03 = "/w04.txt";  static const char *B_OP03 = "/b04.txt";
static const char *W_OP04 = "/w05.txt";  static const char *B_OP04 = "/b05.txt";
static const char *W_OP05 = "/w06.txt";  static const char *B_OP05 = "/b06.txt";
static const char *W_OP06 = "/w07.txt";  static const char *B_OP06 = "/b07.txt";
static const char *W_OP07 = "/w08.txt";  static const char *B_OP07 = "/b08.txt";
static const char *W_OP08 = "/w09.txt";  static const char *B_OP08 = "/b09.txt";
static const char *W_OP09 = "/w10.txt";  static const char *B_OP09 = "/b10.txt";
static const char *W_OP10 = "/w11.txt";  static const char *B_OP10 = "/b11.txt";
static const char *W_OP11 = "/w12.txt";  static const char *B_OP11 = "/b12.txt";
static const char *W_OP12 = "/w13.txt";  static const char *B_OP12 = "/b13.txt";
static const char *W_OP13 = "/w14.txt";  static const char *B_OP13 = "/b14.txt";
static const char *W_OP14 = "/w15.txt";  static const char *B_OP14 = "/b15.txt";
static const char *W_OP15 = "/w16.txt";  static const char *B_OP15 = "/b16.txt";
static const char *W_OP16 = "/w17.txt";  static const char *B_OP16 = "/b17.txt";
static const char *W_OP17 = "/w18.txt";  static const char *B_OP17 = "/b18.txt";
static const char *W_OP18 = "/w19.txt";  static const char *B_OP18 = "/b19.txt";
static const char *W_OP19 = "/w20.txt";  static const char *B_OP19 = "/b20.txt";
static const char *W_OP20 = "/w21.txt";  static const char *B_OP20 = "/b21.txt";
static const char *W_OP21 = "/w22.txt";  static const char *B_OP21 = "/b22.txt";
static const char *W_OP22 = "/w23.txt";  static const char *B_OP22 = "/b23.txt";
static const char *W_OP23 = "/w24.txt";  static const char *B_OP23 = "/b24.txt";
static const char *W_OP24 = "/w25.txt";  static const char *B_OP24 = "/b25.txt";
static const char *W_OP25 = "/w26.txt";  static const char *B_OP25 = "/b26.txt";
static const char *W_OP26 = "/w27.txt";  static const char *B_OP26 = "/b27.txt";

static const char *FC_W = "/w28.txt";
static const char *FC_B = "/b28.txt";

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

// =====================================
// Minimal serial helpers
// =====================================
static bool serial_read_exact(uint8_t *dst, uint32_t n) {
  uint32_t got = 0;
  while (got < n) {
    int r = Serial.readBytes((char*)dst + got, (size_t)(n - got));
    if (r > 0) got += (uint32_t)r;
    else delay(1);
  }
  return true;
}

static uint32_t read_u32_le_blocking() {
  uint8_t b[4];
  serial_read_exact(b, 4);
  return (uint32_t)b[0] |
         ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
}

static void rgb_u8_to_planar_float_0_1(const uint8_t *rgb, float *out_chw, uint16_t W) {
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
static void alloc_buffers() {
  const uint32_t maxTensor = (uint32_t)48 * 48 * 16; // 36864 floats
  A   = (float*)malloc(maxTensor * sizeof(float));
  B   = (float*)malloc(maxTensor * sizeof(float));
  TMP = (float*)malloc((uint32_t)IN_W * (uint32_t)IN_W * sizeof(float));      // 96*96 floats
  IN  = (float*)malloc((uint32_t)IN_W * (uint32_t)IN_W * IN_C * sizeof(float)); // 96*96*3 floats
  RGB = (uint8_t*)malloc(IN_RGB_BYTES); // 27648 bytes
}

// =====================================
// Inference
// =====================================
static void run_vww_on_IN_and_report() {
  Pool POOL_ID; POOL_ID.M = 1; POOL_ID.T = 1;

  uint16_t W = IN_W;
  uint16_t V = 0;

  Conv c00; c00.K=3; c00.P=1; c00.S=2; c00.weight_fn=W_OP00; c00.bias_fn=B_OP00; c00.act=ACT_RELU;
  V = noodle_conv_float(IN, 3, 8, A, W, c00, POOL_ID, nullptr); W = V;

  Conv d01; d01.K=3; d01.P=1; d01.S=1; d01.weight_fn=W_OP01; d01.bias_fn=B_OP01; d01.act=ACT_RELU;
  V = noodle_dwconv_float(A, 8, B, W, d01, POOL_ID, nullptr); W = V;

  Conv c02; c02.K=1; c02.P=0; c02.S=1; c02.weight_fn=W_OP02; c02.bias_fn=B_OP02; c02.act=ACT_RELU;
  V = noodle_conv_float(B, 8, 16, A, W, c02, POOL_ID, nullptr); W = V;

  Conv d03; d03.K=3; d03.P=1; d03.S=2; d03.weight_fn=W_OP03; d03.bias_fn=B_OP03; d03.act=ACT_RELU;
  V = noodle_dwconv_float(A, 16, B, W, d03, POOL_ID, nullptr); W = V;

  Conv c04; c04.K=1; c04.P=0; c04.S=1; c04.weight_fn=W_OP04; c04.bias_fn=B_OP04; c04.act=ACT_RELU;
  V = noodle_conv_float(B, 16, 32, A, W, c04, POOL_ID, nullptr); W = V;

  Conv d05; d05.K=3; d05.P=1; d05.S=1; d05.weight_fn=W_OP05; d05.bias_fn=B_OP05; d05.act=ACT_RELU;
  V = noodle_dwconv_float(A, 32, B, W, d05, POOL_ID, nullptr); W = V;

  Conv c06; c06.K=1; c06.P=0; c06.S=1; c06.weight_fn=W_OP06; c06.bias_fn=B_OP06; c06.act=ACT_RELU;
  V = noodle_conv_float(B, 32, 32, A, W, c06, POOL_ID, nullptr); W = V;

  Conv d07; d07.K=3; d07.P=1; d07.S=2; d07.weight_fn=W_OP07; d07.bias_fn=B_OP07; d07.act=ACT_RELU;
  V = noodle_dwconv_float(A, 32, B, W, d07, POOL_ID, nullptr); W = V;

  Conv c08; c08.K=1; c08.P=0; c08.S=1; c08.weight_fn=W_OP08; c08.bias_fn=B_OP08; c08.act=ACT_RELU;
  V = noodle_conv_float(B, 32, 64, A, W, c08, POOL_ID, nullptr); W = V;

  Conv d09; d09.K=3; d09.P=1; d09.S=1; d09.weight_fn=W_OP09; d09.bias_fn=B_OP09; d09.act=ACT_RELU;
  V = noodle_dwconv_float(A, 64, B, W, d09, POOL_ID, nullptr); W = V;

  Conv c10; c10.K=1; c10.P=0; c10.S=1; c10.weight_fn=W_OP10; c10.bias_fn=B_OP10; c10.act=ACT_RELU;
  V = noodle_conv_float(B, 64, 64, A, W, c10, POOL_ID, nullptr); W = V;

  Conv d11; d11.K=3; d11.P=1; d11.S=2; d11.weight_fn=W_OP11; d11.bias_fn=B_OP11; d11.act=ACT_RELU;
  V = noodle_dwconv_float(A, 64, B, W, d11, POOL_ID, nullptr); W = V;

  Conv c12; c12.K=1; c12.P=0; c12.S=1; c12.weight_fn=W_OP12; c12.bias_fn=B_OP12; c12.act=ACT_RELU;
  V = noodle_conv_float(B, 64, 128, A, W, c12, POOL_ID, nullptr); W = V;

  Conv d13; d13.K=3; d13.P=1; d13.S=1; d13.weight_fn=W_OP13; d13.bias_fn=B_OP13; d13.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d13, POOL_ID, nullptr); W = V;

  Conv c14; c14.K=1; c14.P=0; c14.S=1; c14.weight_fn=W_OP14; c14.bias_fn=B_OP14; c14.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c14, POOL_ID, nullptr); W = V;

  Conv d15; d15.K=3; d15.P=1; d15.S=1; d15.weight_fn=W_OP15; d15.bias_fn=B_OP15; d15.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d15, POOL_ID, nullptr); W = V;

  Conv c16; c16.K=1; c16.P=0; c16.S=1; c16.weight_fn=W_OP16; c16.bias_fn=B_OP16; c16.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c16, POOL_ID, nullptr); W = V;

  Conv d17; d17.K=3; d17.P=1; d17.S=1; d17.weight_fn=W_OP17; d17.bias_fn=B_OP17; d17.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d17, POOL_ID, nullptr); W = V;

  Conv c18; c18.K=1; c18.P=0; c18.S=1; c18.weight_fn=W_OP18; c18.bias_fn=B_OP18; c18.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c18, POOL_ID, nullptr); W = V;

  Conv d19; d19.K=3; d19.P=1; d19.S=1; d19.weight_fn=W_OP19; d19.bias_fn=B_OP19; d19.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d19, POOL_ID, nullptr); W = V;

  Conv c20; c20.K=1; c20.P=0; c20.S=1; c20.weight_fn=W_OP20; c20.bias_fn=B_OP20; c20.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c20, POOL_ID, nullptr); W = V;

  Conv d21; d21.K=3; d21.P=1; d21.S=1; d21.weight_fn=W_OP21; d21.bias_fn=B_OP21; d21.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d21, POOL_ID, nullptr); W = V;

  Conv c22; c22.K=1; c22.P=0; c22.S=1; c22.weight_fn=W_OP22; c22.bias_fn=B_OP22; c22.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 128, A, W, c22, POOL_ID, nullptr); W = V;

  Conv d23; d23.K=3; d23.P=1; d23.S=2; d23.weight_fn=W_OP23; d23.bias_fn=B_OP23; d23.act=ACT_RELU;
  V = noodle_dwconv_float(A, 128, B, W, d23, POOL_ID, nullptr); W = V;

  Conv c24; c24.K=1; c24.P=0; c24.S=1; c24.weight_fn=W_OP24; c24.bias_fn=B_OP24; c24.act=ACT_RELU;
  V = noodle_conv_float(B, 128, 256, A, W, c24, POOL_ID, nullptr); W = V;

  Conv d25; d25.K=3; d25.P=1; d25.S=1; d25.weight_fn=W_OP25; d25.bias_fn=B_OP25; d25.act=ACT_RELU;
  V = noodle_dwconv_float(A, 256, B, W, d25, POOL_ID, nullptr); W = V;

  Conv c26; c26.K=1; c26.P=0; c26.S=1; c26.weight_fn=W_OP26; c26.bias_fn=B_OP26; c26.act=ACT_RELU;
  V = noodle_conv_float(B, 256, 256, A, W, c26, POOL_ID, nullptr); W = V;

  noodle_gap(A, 256, W);

  float out2[2];
  FCNFile fcf; fcf.weight_fn = FC_W; fcf.bias_fn = FC_B; fcf.act = ACT_SOFTMAX;
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

  while (!noodle_sd_init()) {
    Serial.println("FS init failed, retry...");
    delay(500);
  }
  Serial.println("FS OK");

  alloc_buffers();
  if (!A || !B || !TMP || !IN || !RGB) {
    Serial.println("ERR malloc");
    while (1) delay(100);
  }

  noodle_setup_temp_buffers((void*)TMP, (void*)B);

  Serial.println("READY"); // Python can wait for this
}

void loop() {
  // Wait forever for a frame: [u32 length][bytes...]
  uint32_t n = read_u32_le_blocking();

  if (n != IN_RGB_BYTES) {
    // Drain whatever was sent, then complain
    uint32_t to_drop = n;
    static uint8_t junk[64];
    while (to_drop) {
      uint32_t k = (to_drop > sizeof(junk)) ? sizeof(junk) : to_drop;
      serial_read_exact(junk, k);
      to_drop -= k;
    }
    Serial.printf("ERR bad_len=%lu expected=%lu\n", (unsigned long)n, (unsigned long)IN_RGB_BYTES);
    return;
  }

  // Read image bytes
  serial_read_exact(RGB, IN_RGB_BYTES);

  // Convert to planar float [0,1]
  rgb_u8_to_planar_float_0_1(RGB, IN, IN_W);

  // Run inference + report time
  unsigned long t0 = millis();
  run_vww_on_IN_and_report();
  unsigned long t1 = millis();

  // Overwrite ms=0 line by printing a second timing line (keeps code simple)
  // If you prefer single-line output, see note below.
  Serial.printf("time_ms=%lu\n", (unsigned long)(t1 - t0));
}
