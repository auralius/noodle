#include <Arduino.h>
#include <SPI.h>
#include "noodle.h"
#include "noodle_serial.h"

// ============================================================
// Blue Pill LeNet-5 MNIST node
// SRAM-activation version, SD-backed parameters
// ============================================================
//
// SD card on SPI2:
//
//   SD CLK   -> PB13
//   SD MISO  -> PB14
//   SD MOSI  -> PB15
//   SD CS    -> PB12
//   SD 3V3   -> 3.3V
//   SD GND   -> GND
//
// Model:
//
//   Input: 28x28
//   Conv1: 5x5, P=2, S=1, 1 -> 6, ReLU
//   Pool1: 2x2, S=2              -> 14x14x6
//   Conv2: 5x5, P=0, S=1, 6 -> 16, ReLU
//   Pool2: 2x2, S=2              -> 5x5x16
//   Flatten: 400
//   FC1: 400 -> 120, ReLU
//   FC2: 120 -> 84, ReLU
//   FC3: 84 -> 10, Softmax
//
// Tensor placement:
//
//   Parameters: SD files w01.bin ... w05.bin, b01.bin ... b05.bin
//   Activations: SRAM only
//   Temporary conv accumulator: SRAM SCRATCH[28*28]
// ============================================================

static const uint8_t SD_CS = PB12;
SPIClass SPI_SD(PB15, PB14, PB13, PB12);  // SPI2

static const uint8_t LED_PIN = PC13;

// ============================================================
// Image / model settings
// ============================================================

static const uint16_t IMG_W = 28;
static const uint16_t IMG_H = 28;
static const uint16_t IMG_SIZE = IMG_W * IMG_H;

static const uint16_t C1 = 6;
static const uint16_t C2 = 16;
static const uint16_t W1 = 14;  // after Conv1 + Pool1
static const uint16_t W2 = 5;   // after Conv2 + Pool2

static const uint16_t ACT1_SIZE = C1 * W1 * W1;   // 1176 floats
static const uint16_t ACT2_SIZE = C2 * W2 * W2;   // 400 floats

// ============================================================
// Buffers
// ============================================================
//
// Blue Pill has 20 KB SRAM.
//
// IMG     = 784 floats  = 3136 bytes
// SCRATCH = 784 floats  = 3136 bytes
// ACT1    = 1176 floats = 4704 bytes
// ACT2    = 400 floats  = 1600 bytes
// RX      = 784 bytes
//
// Total explicit buffers ~= 13.36 KB.
// ============================================================

static float *IMG     = nullptr;  // input image, later flatten/FC ping-pong buffer
static float *SCRATCH = nullptr;  // conv accumulator plane, used as temp_buff2
static float *ACT1    = nullptr;  // Conv1 output, later FC1 output
static float *ACT2    = nullptr;  // Conv2 output, later logits

static uint8_t RX_BYTES[IMG_SIZE];

// ============================================================
// Helpers
// ============================================================

static void blink(int n) {
  for (int i = 0; i < n; i++) {
    digitalWrite(LED_PIN, LOW);   // Blue Pill LED is usually active LOW
    delay(120);
    digitalWrite(LED_PIN, HIGH);
    delay(120);
  }
}

static void make_conv(Conv &c,
                      uint16_t K,
                      uint16_t P,
                      uint16_t S,
                      const char *w_fn,
                      const char *b_fn,
                      Activation act) {
  c.K = K;
  c.P = P;
  c.S = S;
  c.OP = 0;
  c.weight_fn = w_fn;
  c.bias_fn = b_fn;
  c.act = act;
}

static void make_fcn(FCNFile &f,
                     const char *w_fn,
                     const char *b_fn,
                     Activation act) {
  f.weight_fn = w_fn;
  f.bias_fn = b_fn;
  f.act = act;
}

static void alloc_buffers() {
  IMG     = (float *)malloc((size_t)IMG_SIZE * sizeof(float));
  SCRATCH = (float *)malloc((size_t)IMG_SIZE * sizeof(float));
  ACT1    = (float *)malloc((size_t)ACT1_SIZE * sizeof(float));
  ACT2    = (float *)malloc((size_t)ACT2_SIZE * sizeof(float));

  if (!IMG || !SCRATCH || !ACT1 || !ACT2) {
    Serial.println(F("ERROR: malloc failed"));
    while (true) {
      blink(2);
      delay(500);
    }
  }

  // For SRAM-activation convolution, temp_buff2 is the one-plane
  // convolution accumulator. temp_buff1 is passed for compatibility.
  noodle_setup_temp_buffers(IMG, SCRATCH);
}

static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n) {
  for (size_t i = 0; i < n; i++) {
    dst[i] = (float)src[i];
  }
}

// ============================================================
// Noodle prediction
// ============================================================

static void predict() {
  Pool pool;
  pool.M = 2;
  pool.T = 2;

  Conv cnn1;
  make_conv(cnn1, 5, 2, 1, "w01.bin", "b01.bin", ACT_RELU);

  Conv cnn2;
  make_conv(cnn2, 5, 0, 1, "w02.bin", "b02.bin", ACT_RELU);

  FCNFile fcn1;
  make_fcn(fcn1, "w03.bin", "b03.bin", ACT_RELU);

  FCNFile fcn2;
  make_fcn(fcn2, "w04.bin", "b04.bin", ACT_RELU);

  FCNFile fcn3;
  make_fcn(fcn3, "w05.bin", "b05.bin", ACT_SOFTMAX);

  uint16_t V = 0;

  unsigned long t_all = micros();
  unsigned long t0;

  // Conv1: IMG [1][28][28] -> ACT1 [6][14][14]
  t0 = micros();
  V = noodle_conv_float(IMG, 1, 6,
                        ACT1, IMG_W,
                        cnn1, pool, NULL);
  Serial.print(F("T conv1_mem "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  if (V != W1) {
    Serial.print(F("ERR conv1 V="));
    Serial.println(V);
    return;
  }

  // Conv2: ACT1 [6][14][14] -> ACT2 [16][5][5]
  t0 = micros();
  V = noodle_conv_float(ACT1, 6, 16,
                        ACT2, V,
                        cnn2, pool, NULL);
  Serial.print(F("T conv2_mem "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  if (V != W2) {
    Serial.print(F("ERR conv2 V="));
    Serial.println(V);
    return;
  }

  // Flatten: ACT2 CHW [16][5][5] -> IMG HWC-flat [400]
  t0 = micros();
  V = noodle_flat(ACT2, IMG, V, 16);
  Serial.print(F("T flat_mem "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  if (V != ACT2_SIZE) {
    Serial.print(F("ERR flat V="));
    Serial.println(V);
    return;
  }

  // FC1: IMG[400] -> ACT1[120]
  t0 = micros();
  V = noodle_fcn(IMG, V, 120,
                 ACT1, fcn1, NULL);
  Serial.print(F("T fcn1 "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  if (V != 120) {
    Serial.print(F("ERR fcn1 V="));
    Serial.println(V);
    return;
  }

  // FC2: ACT1[120] -> IMG[84]
  t0 = micros();
  V = noodle_fcn(ACT1, V, 84,
                 IMG, fcn2, NULL);
  Serial.print(F("T fcn2 "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  if (V != 84) {
    Serial.print(F("ERR fcn2 V="));
    Serial.println(V);
    return;
  }

  // FC3: IMG[84] -> ACT2[10], includes softmax by ACT_SOFTMAX
  t0 = micros();
  V = noodle_fcn(IMG, V, 10,
                 ACT2, fcn3, NULL);
  Serial.print(F("T fcn3 "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  if (V != 10) {
    Serial.print(F("ERR fcn3 V="));
    Serial.println(V);
    return;
  }

  const float et = (float)(micros() - t_all) * 1e-6f;

  uint16_t pred = 0;
  float max_val = 0.0f;
  noodle_find_max(ACT2, 10, max_val, pred);

  Serial.print(F("PRED "));
  Serial.print(pred);
  Serial.print(' ');
  Serial.print(et, 4);
  Serial.print(' ');
  Serial.print(max_val, 4);
  Serial.println();
  Serial.flush();
  delay(5);
}

// ============================================================
// Arduino setup / loop
// ============================================================

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  blink(2);

  NoodleSerial::begin(115200);
  NoodleSerial::clear_input();

  Serial.println(F("BOOT Blue Pill Noodle SRAM activation"));

  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);

  Serial.println(F("Before SPI_SD.begin"));
  SPI_SD.begin();

  Serial.println(F("Before noodle_fs_init"));

  if (!noodle_fs_init(SD_CS, SPI_SD, 12)) {
    Serial.println(F("noodle_fs_init FAILED"));
    while (true) {
      blink(3);
      delay(500);
    }
  }

  Serial.println(F("SD mounted OK"));

  alloc_buffers();

  Serial.println(F("Buffers allocated"));
  Serial.println(F("Noodle temp buffers set"));
  Serial.println(F("READY"));
}

void loop() {
  if (!NoodleSerial::wait_for_img_header()) {
    NoodleSerial::print_ready();
    return;
  }

  if (!NoodleSerial::recv_image_chunked(RX_BYTES, IMG_SIZE)) {
    NoodleSerial::print_ready();
    return;
  }

  bytes_to_float_image(RX_BYTES, IMG, IMG_SIZE);

  predict();

  NoodleSerial::print_ready();
}
