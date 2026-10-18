/**
 * Noodle Case 1: INT8 FCN on Arduino Uno R3
 *
 * Network: 256 pixels -> FCN(64, ReLU) -> FCN(10, logits) -> softmax
 * Parameters: SD card (w01i.bin, p01i.bin, w02i.bin, p02i.bin)
 * Activations: 330 bytes SRAM (256 + 64 + 10)
 *
 * Serial: wait for READY, send 256 raw grayscale bytes (0..255).
 * Reply: DBG LOGITS_Q ..., PRED <class> <seconds> <confidence>, MEM ..., READY
 *
 * Copy the four model files to the SD card manually. No weight uploader.
 */

#include <Arduino.h>
#include <math.h>
#include "noodle.h"
#include "noodle_int8.h"
#include "model_quant.h"

static const uint16_t INPUT_SIZE = 16 * 16;
static const uint16_t HIDDEN_SIZE = 64;
static const uint16_t OUTPUT_SIZE = 10;
static const uint32_t RX_TIMEOUT_MS = 3000;
static const uint8_t SD_CS = 10;

// Three tensors share Noodle's SRAM arena; no second image buffer.
static NoodleTensor input;
static NoodleTensor hidden;
static NoodleTensor output;

// Informational only: this is the CURRENT heap-to-stack gap, not a peak.
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

// Read 256 pixels directly into the input tensor (no 256-byte RX array).
static bool receive_image() {
  uint8_t *dst = (uint8_t *)noodle_tensor_data(&input);
  const uint32_t started = millis();
  size_t received = 0;

  while (received < INPUT_SIZE) {
    if ((uint32_t)(millis() - started) > RX_TIMEOUT_MS) return false;
    const int available = Serial.available();
    if (available <= 0) { delay(1); continue; }

    const size_t remaining = INPUT_SIZE - received;
    const size_t count = (size_t)available < remaining
        ? (size_t)available : remaining;
    const int got = Serial.readBytes((char *)(dst + received), count);
    if (got > 0) received += (size_t)got;
  }
  return true;
}

// Pixels arrive as uint8 [0,255]; TFLite expects signed INT8.
// Quantize IN PLACE: q = round(pixel / input_scale) + input_zero_point.
static void quantize_image() {
  uint8_t *pixels = (uint8_t *)noodle_tensor_data(&input);
  for (uint16_t i = 0; i < INPUT_SIZE; ++i) {
    const uint8_t pixel = pixels[i];
    const int8_t q = (int8_t)noodle_quantize_float(
        (float)pixel,
        noodle_export_i8::layer01_input_scale,
        noodle_export_i8::layer01_input_zero_point);
    pixels[i] = (uint8_t)q;
  }
}

// Layer 1: 256 -> 64, ReLU. Each SD metadata record is
// [bias:int32, multiplier:int32, shift:int32] (12 little-endian bytes).
static FCNFile first_layer() {
  FCNFile l;
  l.weight_fn = "w01i.bin";
  l.metadata_fn = "p01i.bin";
  l.O = HIDDEN_SIZE;
  l.act = ACT_RELU;
  l.input_scale = noodle_export_i8::layer01_input_scale;
  l.input_zero_point = noodle_export_i8::layer01_input_zero_point;
  l.output_scale = noodle_export_i8::layer01_output_scale;
  l.output_zero_point = noodle_export_i8::layer01_output_zero_point;
  l.activation_min = noodle_export_i8::layer01_activation_min;
  l.activation_max = noodle_export_i8::layer01_activation_max;
  return l;
}

// Layer 2: 64 -> 10. These outputs are logits, NOT probabilities.
static FCNFile second_layer() {
  FCNFile l;
  l.weight_fn = "w02i.bin";
  l.metadata_fn = "p02i.bin";
  l.O = OUTPUT_SIZE;
  l.act = ACT_NONE;
  l.input_scale = noodle_export_i8::layer02_input_scale;
  l.input_zero_point = noodle_export_i8::layer02_input_zero_point;
  l.output_scale = noodle_export_i8::layer02_output_scale;
  l.output_zero_point = noodle_export_i8::layer02_output_zero_point;
  l.activation_min = noodle_export_i8::layer02_activation_min;
  l.activation_max = noodle_export_i8::layer02_activation_max;
  return l;
}

// Softmax for the winning class. Subtract the maximum for stability;
// common zero points cancel when taking differences between logits.
static float winning_probability(int8_t maximum) {
  const int8_t *q = noodle_tensor_const_data(&output);
  float sum = 0.0f;
  for (uint8_t i = 0; i < OUTPUT_SIZE; ++i)
    sum += expf(((int16_t)q[i] - (int16_t)maximum) * output.scale);
  return 1.0f / sum;
}

static bool infer(uint32_t started_us) {
  uint16_t count;

  // 1. FCN1: SRAM input -> SRAM hidden; parameters streamed from SD.
  // A local scope releases this descriptor before constructing FCN2's.
  {
    const FCNFile fc1 = first_layer();
    count = noodle_fcn(&input, &hidden, fc1);
  }
  if (count != HIDDEN_SIZE) { Serial.println(F("ERR FCN1")); return false; }

  // 2. FCN2: SRAM hidden -> SRAM output; parameters streamed from SD.
  {
    const FCNFile fc2 = second_layer();
    count = noodle_fcn(&hidden, &output, fc2);
  }
  if (count != OUTPUT_SIZE) { Serial.println(F("ERR FCN2")); return false; }

  // 3. Classification: argmax(logits), followed by softmax confidence.
  uint16_t predicted_class = 0;
  int8_t largest_logit = -128;
  noodle_i8_find_max(&output, &predicted_class, &largest_logit);
  const float confidence = winning_probability(largest_logit);

  // 4. Stop timer BEFORE serial output. Includes input quantization.
  const uint32_t elapsed_us = micros() - started_us;

  // Raw INT8 logits let the PC verify every output against its reference.
  const int8_t *q = noodle_tensor_const_data(&output);
  Serial.print(F("DBG LOGITS_Q"));
  for (uint8_t i = 0; i < OUTPUT_SIZE; ++i) {
    Serial.print(' ');
    Serial.print((int)q[i]);
  }
  Serial.println();

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

  noodle_tensor_init(&input);
  noodle_tensor_init(&hidden);
  noodle_tensor_init(&output);

  // Allocate ALL 330 activation bytes BEFORE SD initialization. This is
  // essential on the 2-KB Uno: avoid allocating while SD files are open.
  if (!noodle_i8_tensor_require_vector(&input, INPUT_SIZE,
          noodle_export_i8::layer01_input_scale,
          noodle_export_i8::layer01_input_zero_point)) stop_on_error(F("INPUT_ALLOC"));
  if (!noodle_i8_tensor_require_vector(&hidden, HIDDEN_SIZE,
          noodle_export_i8::layer01_output_scale,
          noodle_export_i8::layer01_output_zero_point)) stop_on_error(F("HIDDEN_ALLOC"));
  if (!noodle_i8_tensor_require_vector(&output, OUTPUT_SIZE,
          noodle_export_i8::layer02_output_scale,
          noodle_export_i8::layer02_output_zero_point)) stop_on_error(F("OUTPUT_ALLOC"));

  if (!noodle_fs_init(SD_CS)) stop_on_error(F("SD_INIT"));

  Serial.println(F("BOOT UNO_FCN16_INT8_CLEAN"));
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

  const uint32_t started_us = micros();
  quantize_image();                       // Step 0: uint8 -> INT8
  if (infer(started_us)) report_memory(F("after_predict"));
  Serial.println(F("READY"));
}
