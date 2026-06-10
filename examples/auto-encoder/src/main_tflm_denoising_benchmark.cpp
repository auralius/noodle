#include <Arduino.h>
#if defined(ARDUINO_ARCH_ESP32)
#include "esp_heap_caps.h"
#endif

// TensorFlow Lite Micro.
// This sketch is written for the Arduino framework on ESP32-class boards.
//
// Library note:
// Use an Arduino-compatible TensorFlow Lite Micro library.
// If your installed library uses slightly different include paths, adjust
// only the include section below.

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Some Arduino TFLM packages do not provide tensorflow/lite/version.h.
// TFLITE_SCHEMA_VERSION is normally 3 for TFLite flatbuffer schema.
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3
#endif

#include "noodle_serial.h"
#include "model_data.h"   // must define: g_model[] and g_model_len

static constexpr uint16_t IMG_W = 28;
static constexpr uint16_t IMG_H = 28;
static constexpr uint16_t IMG_SIZE = IMG_W * IMG_H;

// Optional: set to 1 for extra tensor debug prints.
// Keep 0 for Python plotting/benchmarking.
#ifndef AE_VERBOSE
#define AE_VERBOSE 0
#endif

static uint8_t RX_BYTES[IMG_SIZE];
static uint8_t TX_BYTES[IMG_SIZE];

// ============================================================
// TFLite Micro globals
// ============================================================

// Start large for the beginner benchmark.
// If AllocateTensors() fails, increase this value.
// If it succeeds with large unused memory, students can reduce it later.
static constexpr size_t TENSOR_ARENA_SIZE = 96 * 1024;

static uint8_t *tensor_arena = nullptr;

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;

// AllOpsResolver is beginner-friendly because it avoids missing-op errors.
// After the benchmark works, replace this with MicroMutableOpResolver and
// register only the required ops to reduce flash usage.
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
      (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  m.psram_min =
      (uint32_t)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
  m.psram_largest =
      (uint32_t)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
#endif

  return m;
}

static void print_runtime_memory_pair(const char *tag,
                                      const RuntimeMemorySnapshot &before,
                                      const RuntimeMemorySnapshot &after) {
#if defined(ARDUINO_ARCH_ESP32)
  Serial.printf(
      "MEM %s "
      "heap_before=%u heap_after=%u heap_min_before=%u heap_min_after=%u "
      "heap_largest_before=%u heap_largest_after=%u "
      "psram_before=%u psram_after=%u psram_min_before=%u psram_min_after=%u "
      "psram_largest_before=%u psram_largest_after=%u\n",
      tag,
      (unsigned)before.heap_free,
      (unsigned)after.heap_free,
      (unsigned)before.heap_min,
      (unsigned)after.heap_min,
      (unsigned)before.heap_largest,
      (unsigned)after.heap_largest,
      (unsigned)before.psram_free,
      (unsigned)after.psram_free,
      (unsigned)before.psram_min,
      (unsigned)after.psram_min,
      (unsigned)before.psram_largest,
      (unsigned)after.psram_largest);
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
  tensor_arena = (uint8_t *)alloc_bytes(TENSOR_ARENA_SIZE);

  if (!tensor_arena) {
    Serial.println(F("ERROR tensor_arena allocation failed"));
    while (true) delay(1000);
  }

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERROR model schema=%d expected=%d\n",
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
    Serial.println(F("ERROR AllocateTensors failed"));
    Serial.println(F("Try increasing TENSOR_ARENA_SIZE."));
    while (true) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  print_tensor_info("input", input);
  print_tensor_info("output", output);

  if (!tensor_is_28x28x1_float(input)) {
    Serial.println(F("ERROR input tensor is not float32 [1,28,28,1]"));
    while (true) delay(1000);
  }

  if (!tensor_is_28x28x1_float(output)) {
    Serial.println(F("ERROR output tensor is not float32 [1,28,28,1]"));
    while (true) delay(1000);
  }

  Serial.println(F("TFLite Micro setup OK"));
  Serial.printf("Model bytes: %u\n", (unsigned)g_model_len);
  Serial.printf("Tensor arena bytes: %u\n", (unsigned)TENSOR_ARENA_SIZE);

  // No setup memory print here. Some host scripts expect setup to stay quiet
  // except for READY/protocol lines.
}

// ============================================================
// Process one image
// ============================================================

static void process_one_image() {
  // RX_BYTES -> input tensor.
  bytes_to_float_image_0_1(RX_BYTES, input->data.f, IMG_SIZE);

  // Read memory before/after inference, but print it only after the normal
  // OUT + image-byte protocol is finished. This avoids confusing the sender.
  const RuntimeMemorySnapshot mem_before = read_runtime_memory();

  uint32_t t0 = micros();
  TfLiteStatus status = interpreter->Invoke();
  uint32_t dt = micros() - t0;

  const RuntimeMemorySnapshot mem_after = read_runtime_memory();

  if (status != kTfLiteOk) {
    Serial.println(F("ERR_INVOKE"));
    NoodleSerial::print_ready();
    return;
  }

  // output tensor -> TX_BYTES.
  float_image_to_bytes_0_255(output->data.f, TX_BYTES, IMG_SIZE);

  // Keep the original benchmark protocol first:
  // OUT <time_us>
  // 784 binary image bytes
  NoodleSerial::send_output_image(TX_BYTES, IMG_SIZE, dt);

  // Print memory after the image payload so the Python sender is not waiting
  // for binary bytes while receiving ASCII debug lines.
  print_runtime_memory_pair("inference", mem_before, mem_after);
}

// ============================================================
// Arduino
// ============================================================

void setup() {
  NoodleSerial::begin();
  NoodleSerial::clear_input();

  Serial.println();
  Serial.println(F("BOOT TFLM DENOISING BENCHMARK OK"));

#if defined(ARDUINO_ARCH_ESP32)
  Serial.printf("Flash size: %u\n", ESP.getFlashChipSize());
  Serial.printf("PSRAM found: %s\n", psramFound() ? "YES" : "NO");
  Serial.printf("PSRAM size: %u\n", ESP.getPsramSize());
#endif

  setup_tflm();

  NoodleSerial::print_ready();
}

void loop() {
  // Keep the same host protocol as the Noodle benchmark:
  // READY -> IMG -> RDYIMG -> ACK chunks -> OUT <time_us> -> image bytes.
  if (!NoodleSerial::wait_for_img_header()) {
    NoodleSerial::print_ready();
    delay(20);
    return;
  }

  if (!NoodleSerial::recv_image_chunked(RX_BYTES, IMG_SIZE)) {
    NoodleSerial::print_ready();
    return;
  }

  process_one_image();
}
