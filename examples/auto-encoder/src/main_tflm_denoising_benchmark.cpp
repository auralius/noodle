// TFLite Micro DeepSeq-AE-28 denoising benchmark.
// Clean serial protocol + safe memory query command.
// Target: ESP32 / ESP32-S3 class boards using Arduino framework.
//
// Main image protocol:
//   PC -> ESP32: IMG + 784 uint8 pixels, chunked
//   ESP32 -> PC: OUT <microseconds> + 784 uint8 pixels + READY
//
// Extra memory query protocol, safe between images:
//   PC -> ESP32: MEM
//   ESP32 -> PC: MEM <fields...> + READY
//
// IMPORTANT:
//   Do not print debug/memory text between OUT and the 784-byte binary image.
//   Keep all memory printing in setup or in the separate MEM command.

#include <Arduino.h>

#if defined(ARDUINO_ARCH_ESP32)
#include "esp_heap_caps.h"
#endif

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3
#endif

#include "noodle_serial.h"
#include "model_data.h"   // must define: g_model[] and g_model_len

static constexpr uint16_t IMG_W = 28;
static constexpr uint16_t IMG_H = 28;
static constexpr uint16_t IMG_SIZE = IMG_W * IMG_H;

#ifndef AE_VERBOSE
#define AE_VERBOSE 0
#endif

static uint8_t RX_BYTES[IMG_SIZE];
static uint8_t TX_BYTES[IMG_SIZE];

// ============================================================
// TFLite Micro globals
// ============================================================

// model_data.h reports g_model_len = 42668 bytes.
// Start with 96 KiB arena. Later reduce this to find minimum working arena.
static constexpr size_t TENSOR_ARENA_SIZE = 128 * 1024;

static uint8_t *tensor_arena = nullptr;

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;

static tflite::AllOpsResolver resolver;

// ============================================================
// Runtime memory measurement
// ============================================================

struct RuntimeMemorySnapshot {
  uint32_t heap_free;
  uint32_t heap_min;
  uint32_t heap_largest;
  uint32_t psram_free;
  uint32_t psram_min;
  uint32_t psram_largest;
};

static float fragmentation_percent(uint32_t free_bytes, uint32_t largest_block) {
  if (free_bytes == 0) return 0.0f;

  float frag = 100.0f * (1.0f - ((float)largest_block / (float)free_bytes));

  if (frag < 0.0f) frag = 0.0f;
  if (frag > 100.0f) frag = 100.0f;

  return frag;
}

static RuntimeMemorySnapshot read_runtime_memory() {
  RuntimeMemorySnapshot m{0, 0, 0, 0, 0, 0};

#if defined(ARDUINO_ARCH_ESP32)
  m.heap_free =
      (uint32_t)heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  m.heap_min =
      (uint32_t)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  m.heap_largest =
      (uint32_t)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

  m.psram_free =
      (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  m.psram_min =
      (uint32_t)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  m.psram_largest =
      (uint32_t)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#endif

  return m;
}

static void print_memory_state(const char *tag) {
#if defined(ARDUINO_ARCH_ESP32)
  RuntimeMemorySnapshot m = read_runtime_memory();

  const float heap_frag = fragmentation_percent(m.heap_free, m.heap_largest);
  const float psram_frag = fragmentation_percent(m.psram_free, m.psram_largest);

  Serial.printf(
      "MEM %s arena=%u model=%u "
      "heap=%u heap_min=%u heap_largest=%u heap_frag=%.2f%% "
      "psram=%u psram_min=%u psram_largest=%u psram_frag=%.2f%%\n",
      tag,
      (unsigned)TENSOR_ARENA_SIZE,
      (unsigned)g_model_len,
      (unsigned)m.heap_free,
      (unsigned)m.heap_min,
      (unsigned)m.heap_largest,
      heap_frag,
      (unsigned)m.psram_free,
      (unsigned)m.psram_min,
      (unsigned)m.psram_largest,
      psram_frag);
#else
  Serial.print(F("MEM "));
  Serial.print(tag);
  Serial.println(F(" unsupported"));
#endif
}

// ============================================================
// Allocation helper
// ============================================================

static void *alloc_bytes(size_t n) {
#if defined(ARDUINO_ARCH_ESP32)
  void *p = ps_malloc(n);
  if (p) return p;
#endif
  return malloc(n);
}

// ============================================================
// Utility functions
// ============================================================

static void bytes_to_float_image_0_1(const uint8_t *src, float *dst, size_t n) {
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) {
    dst[i] = (float)src[i] * inv;
  }
}

static void float_image_to_bytes_0_255(const float *src, uint8_t *dst, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float v = src[i];

    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;

    int q = (int)(255.0f * v + 0.5f);

    if (q < 0) q = 0;
    if (q > 255) q = 255;

    dst[i] = (uint8_t)q;
  }
}

static void print_tensor_info(const char *name, const TfLiteTensor *t) {
#if AE_VERBOSE
  Serial.printf("%s type=%d dims=", name, (int)t->type);
  if (t->dims) {
    Serial.print("[");
    for (int i = 0; i < t->dims->size; i++) {
      if (i) Serial.print(", ");
      Serial.print(t->dims->data[i]);
    }
    Serial.print("]");
  }
  Serial.println();
#endif
}

static bool tensor_is_28x28x1_float(const TfLiteTensor *t) {
  if (!t) return false;
  if (t->type != kTfLiteFloat32) return false;
  if (!t->dims) return false;

  // Expected input/output: [1, 28, 28, 1]
  if (t->dims->size != 4) return false;
  if (t->dims->data[0] != 1) return false;
  if (t->dims->data[1] != IMG_H) return false;
  if (t->dims->data[2] != IMG_W) return false;
  if (t->dims->data[3] != 1) return false;

  return true;
}

// ============================================================
// TFLite Micro setup
// ============================================================

static void setup_tflm() {
  print_memory_state("before_arena_alloc");

  tensor_arena = (uint8_t *)alloc_bytes(TENSOR_ARENA_SIZE);

  if (!tensor_arena) {
    Serial.println(F("ERR_ARENA_ALLOC"));
    while (true) delay(1000);
  }

  print_memory_state("after_arena_alloc");

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERR_MODEL_SCHEMA got=%d expected=%d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    while (true) delay(1000);
  }

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      TENSOR_ARENA_SIZE);

  interpreter = &static_interpreter;

  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println(F("ERR_ALLOCATE_TENSORS"));
    Serial.println(F("Try increasing TENSOR_ARENA_SIZE."));
    while (true) delay(1000);
  }

  print_memory_state("after_allocate_tensors");

  input = interpreter->input(0);
  output = interpreter->output(0);

  print_tensor_info("input", input);
  print_tensor_info("output", output);

  if (!tensor_is_28x28x1_float(input)) {
    Serial.println(F("ERR_INPUT_TENSOR_NOT_FLOAT32_1x28x28x1"));
    while (true) delay(1000);
  }

  if (!tensor_is_28x28x1_float(output)) {
    Serial.println(F("ERR_OUTPUT_TENSOR_NOT_FLOAT32_1x28x28x1"));
    while (true) delay(1000);
  }

  Serial.println(F("TFLM setup OK"));
  Serial.printf("Model bytes: %u\n", (unsigned)g_model_len);
  Serial.printf("Tensor arena bytes: %u\n", (unsigned)TENSOR_ARENA_SIZE);
}

// ============================================================
// Process one image
// ============================================================

static void process_one_image() {
  bytes_to_float_image_0_1(RX_BYTES, input->data.f, IMG_SIZE);

  const uint32_t t0 = micros();
  TfLiteStatus status = interpreter->Invoke();
  const uint32_t dt = micros() - t0;

  if (status != kTfLiteOk) {
    Serial.println(F("ERR_INVOKE"));
    NoodleSerial::print_ready();
    return;
  }

  float_image_to_bytes_0_255(output->data.f, TX_BYTES, IMG_SIZE);

  // No debug prints after this point.
  // The sender expects OUT line followed immediately by exactly 784 binary bytes.
  NoodleSerial::send_output_image(TX_BYTES, IMG_SIZE, dt);
}

// ============================================================
// Command dispatcher
// ============================================================

static bool read_exact_command3(char cmd[4]) {
  cmd[0] = 0;
  cmd[1] = 0;
  cmd[2] = 0;
  cmd[3] = 0;

  // Non-blocking command read.
  // Only consume bytes when a complete 3-byte command is already available.
  if (Serial.available() < 3) return false;

  const size_t n = Serial.readBytes(cmd, 3);
  return n == 3;
}

static bool cmd_is(const char cmd[4], char a, char b, char c) {
  return cmd[0] == a && cmd[1] == b && cmd[2] == c;
}

// ============================================================
// Arduino
// ============================================================

void setup() {
  NoodleSerial::begin();
  NoodleSerial::clear_input();

  Serial.println();
  Serial.println(F("BOOT TFLM DEEPSEQ_AE28 MEM"));

#if defined(ARDUINO_ARCH_ESP32)
  Serial.printf("Flash size: %u\n", ESP.getFlashChipSize());
  Serial.printf("PSRAM found: %s\n", psramFound() ? "YES" : "NO");
  Serial.printf("PSRAM size: %u\n", ESP.getPsramSize());
#endif

  print_memory_state("boot");

  setup_tflm();

  print_memory_state("ready");

  NoodleSerial::print_ready();
}

void loop() {
  // Supported commands:
  //   IMG : followed by chunked 784-byte image payload
  //   MEM : text-only memory query, safe between images
  char cmd[4];

  if (!read_exact_command3(cmd)) {
    // Stay quiet while idle. Do not spam READY.
    delay(5);
    return;
  }

  if (cmd_is(cmd, 'M', 'E', 'M')) {
    print_memory_state("query");
    NoodleSerial::print_ready();
    return;
  }

  if (cmd_is(cmd, 'I', 'M', 'G')) {
    // Header has already been consumed here, so directly receive chunks.
    if (!NoodleSerial::recv_image_chunked(RX_BYTES, IMG_SIZE)) {
      NoodleSerial::print_ready();
      return;
    }

    process_one_image();
    return;
  }

  Serial.printf("ERR_BAD_CMD %c%c%c\n", cmd[0], cmd[1], cmd[2]);
  NoodleSerial::clear_input();
  NoodleSerial::print_ready();
}
