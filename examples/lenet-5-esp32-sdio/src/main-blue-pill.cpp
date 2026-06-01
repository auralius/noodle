#include <Arduino.h>
#include <SPI.h>
#include "noodle.h"
#include "noodle_serial.h"

// ============================================================
// Blue Pill inference node
// SD card on SPI2
// ============================================================
//
// SD wiring:
//   SD CLK   -> PB13
//   SD MISO  -> PB14
//   SD MOSI  -> PB15
//   SD CS    -> PB12
//   SD 3V3   -> 3.3V
//   SD GND   -> GND
//
// PlatformIO STM32duino SPIClass constructor:
//   SPIClass(MOSI, MISO, SCK, SS)
// ============================================================

static const uint8_t SD_CS = PB12;
SPIClass SPI_SD(PB15, PB14, PB13, PB12);  // SPI2

// Optional heartbeat/debug LED
static const uint8_t LED_PIN = PC13;

// ============================================================
// Image settings
// ============================================================

static const int IMG_W = 28;
static const int IMG_H = 28;
static const int IMG_SIZE = IMG_W * IMG_H;

// ============================================================
// Buffers
// ============================================================
//
// Blue Pill has 20 KB SRAM.
// GRID/BUFFER1 = 784 floats = 3136 bytes
// BUFFER2      = 784 floats = 3136 bytes
// RX_BYTES     = 784 bytes
// ============================================================

float *GRID = nullptr;
float *BUFFER1 = nullptr;
float *BUFFER2 = nullptr;

static uint8_t RX_BYTES[IMG_SIZE];

// ============================================================
// Helpers
// ============================================================

static void blink(int n)
{
  for (int i = 0; i < n; i++) {
    digitalWrite(LED_PIN, LOW);   // Blue Pill LED is usually active LOW
    delay(120);
    digitalWrite(LED_PIN, HIGH);
    delay(120);
  }
}

void alloc_buffers()
{
  GRID    = (float *)malloc((size_t)IMG_SIZE * sizeof(float));
  BUFFER1 = GRID;
  BUFFER2 = (float *)malloc((size_t)IMG_SIZE * sizeof(float));

  if (!BUFFER1 || !BUFFER2) {
    Serial.println(F("ERROR: malloc failed"));
    while (true) {
      blink(2);
      delay(500);
    }
  }
}

static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    dst[i] = (float)src[i];
  }
}

// ============================================================
// Noodle prediction
// ============================================================

void predict()
{
  Conv cnn1;
  cnn1.K = 5;
  cnn1.P = 2;
  cnn1.S = 1;
  cnn1.weight_fn = "w01.bin";
  cnn1.bias_fn   = "b01.bin";

  Conv cnn2;
  cnn2.K = 5;
  cnn2.P = 0;
  cnn2.S = 1;
  cnn2.weight_fn = "w02.bin";
  cnn2.bias_fn   = "b02.bin";

  Pool pool;
  pool.M = 2;
  pool.T = 2;

  FCNFile fcn1;
  fcn1.weight_fn = "w03.bin";
  fcn1.bias_fn   = "b03.bin";
  fcn1.act       = ACT_RELU;

  FCNFile fcn2;
  fcn2.weight_fn = "w04.bin";
  fcn2.bias_fn   = "b04.bin";
  fcn2.act       = ACT_RELU;

  FCNFile fcn3;
  fcn3.weight_fn = "w05.bin";
  fcn3.bias_fn   = "b05.bin";
  fcn3.act       = ACT_SOFTMAX;

  uint16_t V;

  unsigned long t_all = micros();
  unsigned long t0;

  t0 = micros();
  V = noodle_conv_float(GRID, 1, 6,
                        "out1.bin", 28,
                        cnn1, pool, NULL);
  Serial.print(F("T conv1 "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  t0 = micros();
  V = noodle_conv_float("out1.bin", 6, 16,
                        "out2.bin", V,
                        cnn2, pool, NULL);
  Serial.print(F("T conv2 "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  t0 = micros();
  V = noodle_flat("out2.bin", BUFFER1, V, 16);
  Serial.print(F("T flat "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  t0 = micros();
  V = noodle_fcn(BUFFER1, V, 120,
               BUFFER2, fcn1, NULL);
  Serial.print(F("T fcn1 "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  t0 = micros();
  V = noodle_fcn(BUFFER2, V, 84,
               BUFFER1, fcn2, NULL);
  Serial.print(F("T fcn2 "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  t0 = micros();
  V = noodle_fcn(BUFFER1, V, 10,
               BUFFER2, fcn3, NULL);
  Serial.print(F("T fcn3 "));
  Serial.println((micros() - t0) * 1e-6f, 4);

  float et = (float)(micros() - t_all) * 1e-6f;

  uint16_t pred = 0;
  float max_val = 0.0f;
  noodle_find_max(BUFFER2, 10, max_val, pred);

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

void setup()
{
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  blink(2);

  NoodleSerial::begin(115200);
  NoodleSerial::clear_input();

  Serial.println(F("BOOT Blue Pill Noodle SD"));

  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);

  Serial.println(F("Before SPI_SD.begin"));
  SPI_SD.begin();

  Serial.println(F("Before noodle_fs_init"));

  // bool noodle_fs_init(uint8_t cs_pin, SPIClass &spi, uint8_t sck_mhz)
  if (!noodle_fs_init(SD_CS, SPI_SD, 12)) { // 12 MHz seems to be the max for SD cards on Blue Pill
    Serial.println(F("noodle_fs_init FAILED"));
    while (true) {
      blink(3);
      delay(500);
    }
  }

  Serial.println(F("SD mounted OK"));

  alloc_buffers();

  Serial.println(F("Buffers allocated"));

  noodle_setup_temp_buffers(BUFFER1, BUFFER2);

  Serial.println(F("Noodle temp buffers set"));
  Serial.println(F("READY"));
}

void loop()
{
  if (!NoodleSerial::wait_for_img_header()) {
    NoodleSerial::print_ready();
    return;
  }

  if (!NoodleSerial::recv_image_chunked(RX_BYTES, IMG_SIZE)) {
    NoodleSerial::print_ready();
    return;
  }

  bytes_to_float_image(RX_BYTES, BUFFER1, IMG_SIZE);

  predict();

  NoodleSerial::print_ready();
}