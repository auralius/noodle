#include <Arduino.h>
#include "noodle.h"

// -----------------------------
// Waveshare ESP32-P4-Pico SDMMC pins (from Waveshare wiki)
// -----------------------------
static const int SD_CLK = 43;
static const int SD_CMD = 44;
static const int SD_D0  = 39;
static const int SD_D1  = 40;
static const int SD_D2  = 41;
static const int SD_D3  = 42;

// -----------------------------
// Serial RX protocol settings
// -----------------------------
static const uint32_t BAUD = 115200;
static const uint32_t RX_TIMEOUT_MS = 1000;   // timeout waiting for full frame
static const int IMG_W = 28;
static const int IMG_H = 28;
static const int IMG_SIZE = IMG_W * IMG_H;


// -----------------------------
// Buffers
// -----------------------------
float *GRID;
float *BUFFER1;
float *BUFFER2;

static uint8_t RX_BYTES[IMG_SIZE];

// Forward decl
static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms);
static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n);
static int argmax(const float *v, int n);

void alloc_buffers()
{
  // Input image as float (BUFFER1)
  GRID    = (float *)malloc(IMG_SIZE * sizeof(float));
  BUFFER1 = GRID;

  // Temp buffer for Noodle internals (user provided)
  BUFFER2 = (float *)malloc(IMG_SIZE * sizeof(float));

  if (!BUFFER1 || !BUFFER2) {
    Serial.println(F("ERROR: malloc failed (out of RAM)"));
    while (true) delay(1000);
  }
}

void predict()
{
  //noodle_grid_to_file(GRID, "in1.txt", 28);

  Conv cnn1;
  cnn1.K = 5;
  cnn1.P = 2;
  cnn1.S = 1; // same padding
  cnn1.weight_fn = "w01.txt";
  cnn1.bias_fn   = "b01.txt";

  Conv cnn2;
  cnn2.K = 5;
  cnn2.P = 0;
  cnn2.S = 1; // valid padding
  cnn2.weight_fn = "w02.txt";
  cnn2.bias_fn   = "b02.txt";

  Pool pool;
  pool.M = 2;
  pool.T = 2;

  FCNFile fcn1;
  fcn1.weight_fn = "w03.txt";
  fcn1.bias_fn   = "b03.txt";
  fcn1.act    = ACT_RELU;

  FCNFile fcn2;
  fcn2.weight_fn = "w04.txt";
  fcn2.bias_fn   = "b04.txt";
  fcn2.act    = ACT_RELU;

  FCNFile fcn3;
  fcn3.weight_fn = "w05.txt";
  fcn3.bias_fn   = "b05.txt";
  fcn3.act    = ACT_SOFTMAX;

  unsigned long st = micros();
  uint16_t V;

  //V = noodle_conv_float("in1.txt", 1, 6, "out1.txt", 28, cnn1, pool, NULL);
  V = noodle_conv_float(GRID, 1, 6, "out1.txt", 28, cnn1, pool, NULL);
  V = noodle_conv_float("out1.txt", 6, 16, "out2.txt", V, cnn2, pool, NULL);
  V = noodle_flat("out2.txt", BUFFER1, V, 16);

  V = noodle_fcn(BUFFER1, V, 120, "out3.txt", fcn1, NULL);
  V = noodle_fcn("out3.txt", V, 84,  "out4.txt", fcn2, NULL);
  V = noodle_fcn("out4.txt", V, 10,  BUFFER2, fcn3, NULL);
  
  float et = (float)(micros() - st) * 1e-6f;

  // BUFFER1 now holds 10 softmax 
  uint16_t pred = 0;
  float max_val;
  noodle_find_max(BUFFER2, 10, max_val, pred);

  
  // Python-friendly single-line response:
  // PRED <digit> <seconds> <p0> <p1> ... <p9>
  Serial.print("PRED ");
  Serial.print(pred);
  Serial.print(' ');
  Serial.print(et, 4);
  Serial.print(' ');
  Serial.print(max_val, 4);
  Serial.println();
}

void setup()
{
  Serial.begin(BAUD);
  // Clear any garbage from boot
  delay(1000);
  while (Serial.available()) Serial.read();
  
  // Use the NEW 4-bit init overload
  //if (!noodle_fs_init_idf(SD_CLK, SD_CMD, SD_D0, SD_D1, SD_D2, SD_D3)) {
  if (!noodle_fs_init(SD_CLK, SD_CMD, SD_D0)) {
    Serial.println("noodle_fs_init FAILED");
    while (1) delay(1000);
  }
  Serial.println("SDMMC mounted OK");
    
  alloc_buffers();
  noodle_setup_temp_buffers(BUFFER1, BUFFER2);

  // Tell Python we are alive
  Serial.println(F("READY"));
}

void loop()
{
  // 1) Receive 784 bytes
  if (!recv_exact(RX_BYTES, IMG_SIZE, RX_TIMEOUT_MS)) {
    Serial.println(F("READY"));
    return;
  }

  // 2) Convert to float image input
  bytes_to_float_image(RX_BYTES, BUFFER1, IMG_SIZE);

  // 3) Run inference + print result line
  predict();
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms)
{
  uint32_t t_last = millis();
  size_t got = 0;

  while (got < n) {
    if ((millis() - t_last) > timeout_ms) return false;

    int c = Serial.read();
    if (c < 0) { delay(1); continue; }

    dst[got++] = (uint8_t)c;
    t_last = millis(); // reset timeout on progress
  }
  return true;
}


static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n)
{
  // Keep “byte-ness” but stored as float: 0..255
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i];
}

static int argmax(const float *v, int n)
{
  int mi = 0;
  float mv = v[0];
  for (int i = 1; i < n; i++) {
    if (v[i] > mv) { mv = v[i]; mi = i; }
  }
  return mi;
}
