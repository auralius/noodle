/**
 * Noodle Case 1: FP32 FCN on Arduino Uno R3
 *
 * Network: 16x16 uint8 pixels -> FCN(64, ReLU) -> FCN(10, Softmax)
 * Parameters: SD card (w01f.bin, b01f.bin, w02f.bin, b02f.bin)
 * Activations: uint8 input in SRAM; FP32 hidden/output in Noodle SRAM arena
 *
 * Serial: wait for READY, send 256 raw grayscale bytes (0..255).
 * Reply: PRED <class> <seconds> <confidence>, MEM ..., READY
 *
 * Model files are copied to the SD card manually. No weight uploader.
 */

// Preserve the original FP32 example's input convention.
#define NORMALIZE_0_1
#include <Arduino.h>
#include "noodle.h"

static const uint16_t INPUT_SIZE  = 16 * 16;
static const uint16_t HIDDEN_SIZE = 64;
static const uint16_t OUTPUT_SIZE = 10;
static const uint32_t RX_TIMEOUT_MS = 3000;
static const uint8_t SD_CS = 10;

// Input stays as 256 raw bytes; Noodle holds the float activations in SRAM.
static uint8_t pixels[INPUT_SIZE];
static NoodleBuffer hidden;
static NoodleBuffer output;

// Informational: CURRENT stack-to-heap gap, not peak stack headroom.
extern int __heap_start;
extern void *__brkval;

static int free_ram() {
  int stack_marker;
  const uintptr_t heap_end = __brkval
      ? (uintptr_t)__brkval : (uintptr_t)&__heap_start;
  return (int)((uintptr_t)&stack_marker - heap_end);
}

static void report_memory(const __FlashStringHelper *when) {
  Serial.print(F("MEM "));
  Serial.print(when);
  Serial.print(F(" I_bytes="));
  Serial.print(sizeof(pixels));
  Serial.print(F(" A_bytes="));
  Serial.print(noodle_buffer_capacity_bytes(&hidden));
  Serial.print(F(" B_bytes="));
  Serial.print(noodle_buffer_capacity_bytes(&output));
  Serial.print(F(" arena_capacity="));
  Serial.print(noodle_buffer_arena_capacity_bytes());
  Serial.print(F(" arena_used="));
  Serial.print(noodle_buffer_arena_used_bytes());
  Serial.print(F(" free_ram="));
  Serial.println(free_ram());
}

static void stop_on_error(const __FlashStringHelper *message) {
  Serial.print(F("ERR "));
  Serial.println(message);
  while (true) delay(1000);
}

// Receive exactly one image, with no extra temporary buffer.
static bool receive_image() {
  const uint32_t started = millis();
  size_t received = 0;

  while (received < INPUT_SIZE) {
    if ((uint32_t)(millis() - started) > RX_TIMEOUT_MS) return false;
    const int available = Serial.available();
    if (available <= 0) { delay(1); continue; }

    const size_t remaining = INPUT_SIZE - received;
    const size_t count = (size_t)available < remaining
        ? (size_t)available : remaining;
    const int got = Serial.readBytes((char *)(pixels + received), count);
    if (got > 0) received += (size_t)got;
  }
  return true;
}

// FCN1: 256 pixels -> 64 hidden neurons, with ReLU.
static FCNFile first_layer() {
  FCNFile layer{};
  layer.weight_fn = "w01f.bin";
  layer.bias_fn = "b01f.bin";
  layer.act = ACT_RELU;
  return layer;
}

// FCN2: 64 hidden neurons -> 10 class probabilities (Softmax).
static FCNFile second_layer() {
  FCNFile layer{};
  layer.weight_fn = "w02f.bin";
  layer.bias_fn = "b02f.bin";
  layer.act = ACT_SOFTMAX;
  return layer;
}

static bool infer() {
  // Start before the first layer, as in the INT8 Case 1 benchmark.
  const uint32_t started_us = micros();
  uint16_t count;

  // 1. FCN1: raw uint8 input -> FP32 hidden activations (ReLU).
  {
    const FCNFile fc1 = first_layer();
    count = noodle_fcn(pixels, INPUT_SIZE, HIDDEN_SIZE, &hidden, fc1, nullptr);
  }
  if (count != HIDDEN_SIZE) {
    Serial.println(F("ERR FCN1"));
    return false;
  }

  // 2. FCN2: FP32 hidden -> FP32 output probabilities (Softmax).
  {
    const FCNFile fc2 = second_layer();
    count = noodle_fcn(&hidden, HIDDEN_SIZE, OUTPUT_SIZE, &output, fc2, nullptr);
  }
  if (count != OUTPUT_SIZE) {
    Serial.println(F("ERR FCN2"));
    return false;
  }

  // 3. Find the winning class and its Softmax probability.
  uint16_t predicted_class = 0;
  float confidence = 0.0f;
  noodle_find_max(&output, OUTPUT_SIZE, confidence, predicted_class);

  // 4. Stop before serial output: processing + FCNs + classification.
  const uint32_t elapsed_us = micros() - started_us;

  Serial.print(F("PRED "));
  Serial.print(predicted_class);
  Serial.print(' ');
  Serial.print((float)elapsed_us * 1.0e-6f, 4);
  Serial.print(' ');
  Serial.println(confidence, 6);
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  delay(300);
  while (Serial.available()) Serial.read();

  noodle_buffer_init(&hidden);
  noodle_buffer_init(&output);

  // Allocate BOTH FP32 activation buffers before opening any SD file.
  // This avoids growing the SRAM arena inside the filesystem call stack.
  if (!noodle_buffer_require(&hidden, HIDDEN_SIZE))
    stop_on_error(F("HIDDEN_ALLOC"));
  if (!noodle_buffer_require(&output, OUTPUT_SIZE))
    stop_on_error(F("OUTPUT_ALLOC"));

  if (!noodle_fs_init(SD_CS)) stop_on_error(F("SD_INIT"));

  Serial.println(F("BOOT UNO_FCN16_FP32_CLEAN"));
  report_memory(F("boot"));
  Serial.println(F("READY"));
}

void loop() {
  if (!Serial.available()) return;
  if (!receive_image()) {
    Serial.println(F("ERR RX_TIMEOUT"));
    Serial.println(F("READY"));
    return;
  }

  if (infer()) report_memory(F("after_predict"));
  Serial.println(F("READY"));
}
