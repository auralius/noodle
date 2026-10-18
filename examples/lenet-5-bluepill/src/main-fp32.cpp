/*
 * Noodle Case Study 3 — LeNet-5 FP32 using NoodleTensor on Blue Pill
 *
 * Uses the SAME test/blue-pill-sender.py as the INT8 build:
 *     IMG -> RDYIMG -> 784 raw uint8 pixels / 64-byte ACK chunks
 *         -> PRED <class> <seconds> <confidence> -> CRC32 -> STAGE -> READY
 *
 * SD: SPI2, PB13=SCK PB14=MISO PB15=MOSI PB12=CS, requested 12 MHz.
 * SD files: w01f.bin..w05f.bin and b01f.bin..b05f.bin (IEEE-754 FP32).
 * All intermediate activations and convolution scratch remain in SRAM.
 * Cleanup experiment: NOT the original paper timing/memory benchmark firmware.
 *
 * Model: conv5x5 SAME 1->6 ReLU -> meanpool2 -> conv5x5 VALID 6->16
 *        ReLU -> meanpool2 -> CHW-to-HWC flatten (400) -> FC120 ReLU
 *        -> FC84 ReLU -> FC10 -> softmax.
 * Input values are raw uint8 0..255 converted to float; no normalization.
 */

#include <Arduino.h>
#include <SPI.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "noodle.h"
#include "noodle_serial.h"

#if !defined(NOODLE_USE_FLOAT32) || defined(NOODLE_USE_INT8)
#error "Build main-fp32.cpp with -D NOODLE_USE_FLOAT32 (not NOODLE_USE_INT8)"
#endif
#if !defined(NOODLE_USE_SDFAT)
#error "The Case 3 Blue Pill FP32 deployment requires SdFat"
#endif

namespace {
constexpr uint8_t SD_CS = PB12;
constexpr uint8_t LED_PIN = PC13;
constexpr uint16_t IMG_SIDE = 28;
constexpr uint16_t IMG_SIZE = IMG_SIDE * IMG_SIDE;
static SPIClass SPI_SD(PB15, PB14, PB13, PB12);
static uint8_t RX_BYTES[IMG_SIZE];

// Three persistent NoodleTensors share the grow-only NoodleBuffer arena.
// The same physical buffers are reused across the network:
//   IMG  : input 1x28x28 -> flattened 400 -> FC2 output 84
//   ACT1 : conv1/pool 6x14x14 -> FC1 output 120
//   ACT2 : conv2/pool 16x5x5 -> logits 10 -> softmax
// Noodle internally allocates the single 28x28 FP32 conv scratch plane.
// Do not create seven simultaneous FP32 tensors on this 20 KiB MCU.
static NoodleTensor IMG, ACT1, ACT2;

static const char *const WF[5] = {"w01f.bin", "w02f.bin", "w03f.bin", "w04f.bin", "w05f.bin"};
static const char *const BF[5] = {"b01f.bin", "b02f.bin", "b03f.bin", "b04f.bin", "b05f.bin"};
static const uint32_t WEIGHTS[5] = {150, 2400, 48000, 10080, 840};
static const uint16_t OUTPUTS[5] = {6, 16, 120, 84, 10};
static const char *const STAGE_NAMES[7] = {
  "conv1_pool1", "conv2_pool2", "flatten", "fc1", "fc2", "fc3", "softmax"
};

static void blink(uint8_t count) {
  for (uint8_t i = 0; i < count; ++i) {
    digitalWrite(LED_PIN, LOW); delay(120);
    digitalWrite(LED_PIN, HIGH); delay(120);
  }
}

static void boot_fail(const char *reason) {
  Serial.print(F("ERR BOOT ")); Serial.println(reason);
  Serial.flush();
  while (true) { blink(3); delay(500); }
}

static bool file_size_matches(const char *name, uint32_t expected_bytes) {
  NDL_File f = noodle_fs_open_read(name);
  if (!f) {
    Serial.print(F("ERR FILE_MISSING ")); Serial.println(name);
    return false;
  }
  const uint32_t actual_bytes = (uint32_t)f.size();
  f.close();
  if (actual_bytes != expected_bytes) {
    Serial.print(F("ERR FILE_SIZE ")); Serial.print(name);
    Serial.print(' '); Serial.print(actual_bytes);
    Serial.print(F(" expected=")); Serial.println(expected_bytes);
    return false;
  }
  return true;
}

static bool check_fp32_files() {
  // Refuse an INT8 SD deployment: the names overlap but sizes differ.
  for (uint8_t i = 0; i < 5; ++i) {
    if (!file_size_matches(WF[i], WEIGHTS[i] * sizeof(float)) ||
        !file_size_matches(BF[i], (uint32_t)OUTPUTS[i] * sizeof(float)))
      return false;
  }
  return true;
}

static uint32_t crc32_pixels(const uint8_t *data, uint16_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (uint16_t i = 0; i < n; ++i) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ ((crc & 1u) ? 0xEDB88320u : 0u);
  }
  return ~crc;
}

static void print_crc32(uint32_t crc) {
  Serial.print(F("CRC32 "));
  for (int8_t shift = 28; shift >= 0; shift -= 4) {
    const uint8_t digit = (uint8_t)((crc >> shift) & 0xFu);
    Serial.print((char)(digit < 10 ? '0' + digit : 'A' + digit - 10));
  }
  Serial.println();
}

static bool allocate_tensors() {
  noodle_tensor_init(&IMG);
  noodle_tensor_init(&ACT1);
  noodle_tensor_init(&ACT2);

  // Pre-grow to the three peak activation capacities. Keeping these tensors
  // registered across predictions avoids repeated arena allocation/free.
  return noodle_tensor_require_2d(&IMG, 1, IMG_SIDE) != nullptr &&
         noodle_tensor_require_2d(&ACT1, 6, 14) != nullptr &&
         noodle_tensor_require_2d(&ACT2, 16, 5) != nullptr;
}

static bool checked_shape(const NoodleTensor &t,
                          uint8_t rank, uint16_t channels, uint16_t width) {
  return t.rank == rank && t.C == channels && t.W == width &&
         noodle_tensor_const_data(&t) != nullptr;
}

static Conv make_conv(uint16_t pad, uint8_t layer) {
  Conv c;
  c.K = 5; c.P = pad; c.S = 1; c.OP = 0;
  c.O = OUTPUTS[layer];  // required by NoodleTensor wrapper
  c.weight_fn = WF[layer]; c.bias_fn = BF[layer];
  c.act = ACT_RELU;
  return c;
}

static FCNFile make_fc(uint8_t layer) {
  FCNFile f;
  f.O = OUTPUTS[layer];  // required by NoodleTensor wrapper
  f.weight_fn = WF[layer]; f.bias_fn = BF[layer];
  f.act = (layer == 4) ? ACT_NONE : ACT_RELU;
  return f;
}

static void predict() {
  uint32_t stage_us[7] = {0};
  float raw_logits[10] = {0};
  const uint32_t crc = crc32_pixels(RX_BYTES, IMG_SIZE);
  const uint32_t all_start = micros();
  uint16_t width = 0;
  uint32_t t0 = 0;
  uint8_t failure = 0;

  // IMG was reused as a rank-1 FCN output during the preceding prediction.
  // Reestablish its input-image shape without reallocating its capacity.
  float *pixels = noodle_tensor_require_2d(&IMG, 1, IMG_SIDE);
  if (!pixels) { failure = 1; goto done; }
  for (uint16_t i = 0; i < IMG_SIZE; ++i) pixels[i] = (float)RX_BYTES[i];

  {
    Pool pool;
    pool.M = 2; pool.T = 2;

    t0 = micros();
    width = noodle_conv2d(&IMG, &ACT1, make_conv(2, 0), pool);
    stage_us[0] = micros() - t0;
    if (width != 14 || !checked_shape(ACT1, NOODLE_TENSOR_2D, 6, 14)) {
      failure = 2; goto done;
    }

    t0 = micros();
    width = noodle_conv2d(&ACT1, &ACT2, make_conv(0, 1), pool);
    stage_us[1] = micros() - t0;
    if (width != 5 || !checked_shape(ACT2, NOODLE_TENSOR_2D, 16, 5)) {
      failure = 3; goto done;
    }
  }

  // Flatten into the no-longer-needed image buffer; output is 400 floats.
  t0 = micros();
  width = noodle_flat(&ACT2, &IMG);
  stage_us[2] = micros() - t0;
  if (width != 400 || !checked_shape(IMG, NOODLE_TENSOR_1D, 400, 1)) {
    failure = 4; goto done;
  }

  // Reuse large convolution buffers as smaller dense output vectors.
  t0 = micros();
  width = noodle_fcn(&IMG, &ACT1, make_fc(2));
  stage_us[3] = micros() - t0;
  if (width != 120 || !checked_shape(ACT1, NOODLE_TENSOR_1D, 120, 1)) {
    failure = 5; goto done;
  }

  t0 = micros();
  width = noodle_fcn(&ACT1, &IMG, make_fc(3));
  stage_us[4] = micros() - t0;
  if (width != 84 || !checked_shape(IMG, NOODLE_TENSOR_1D, 84, 1)) {
    failure = 6; goto done;
  }

  t0 = micros();
  width = noodle_fcn(&IMG, &ACT2, make_fc(4));
  stage_us[5] = micros() - t0;
  if (width != 10 || !checked_shape(ACT2, NOODLE_TENSOR_1D, 10, 1)) {
    failure = 7; goto done;
  }

  // Preserve raw pre-softmax logits for comparison with existing sender.
  memcpy(raw_logits, noodle_tensor_const_data(&ACT2), sizeof(raw_logits));

  t0 = micros();
  width = noodle_soft_max(&ACT2);
  stage_us[6] = micros() - t0;
  if (width != 10 || !checked_shape(ACT2, NOODLE_TENSOR_1D, 10, 1)) {
    failure = 8; goto done;
  }

  {
    const float *probs = noodle_tensor_const_data(&ACT2);
    float confidence = 0.0f;
    uint16_t pred = 0;
    noodle_find_max(noodle_tensor_buffer(&ACT2), 10, confidence, pred);
    const uint32_t total_us = micros() - all_start;
    Serial.print(F("PRED ")); Serial.print(pred);
    Serial.print(' '); Serial.print(total_us * 1.0e-6f, 4);
    Serial.print(' '); Serial.println(confidence, 6);
    print_crc32(crc);
    Serial.print(F("LOGITS_F"));
    for (uint8_t i = 0; i < 10; ++i) {
      Serial.print(' '); Serial.print(raw_logits[i], 7);
    }
    Serial.println();
    Serial.print(F("SOFTMAX_F"));
    for (uint8_t i = 0; i < 10; ++i) {
      Serial.print(' '); Serial.print(probs[i], 7);
    }
    Serial.println();
    for (uint8_t i = 0; i < 7; ++i) {
      Serial.print(F("STAGE ")); Serial.print(STAGE_NAMES[i]);
      Serial.print(' '); Serial.println(stage_us[i]);
    }
  }

done:
  if (failure) {
    Serial.print(F("ERR CASE3_FP32_STAGE_")); Serial.println(failure);
  }
  Serial.flush();
  delay(5);
}

} // namespace

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  blink(2);
  NoodleSerial::begin(115200);
  NoodleSerial::clear_input();
  Serial.println(F("BOOT Blue Pill Noodle LeNet5 FP32"));
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);
  SPI_SD.begin();
  if (!noodle_fs_init(SD_CS, SPI_SD, 12)) boot_fail("SD_INIT");
  Serial.println(F("SD mounted OK"));
  if (!check_fp32_files()) boot_fail("WRONG_FP32_SD_FILES");
  Serial.println(F("FP32 SD files: 10/10 OK"));
  if (!allocate_tensors()) boot_fail("SRAM_ALLOC");
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
  predict();
  NoodleSerial::print_ready();
}
