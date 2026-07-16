// ============================================================
// TFLite Micro SqueezeNet-1.1 224x224 benchmark for ESP32-S3
// Target: ESP32-S3 N16R8 / N32R16 with PSRAM
//
// This version uses the same NoodleSerial implementation as the
// Noodle SqueezeNet benchmark:
//
//   MCU prints READY
//   Host sends b"IMG"
//   MCU prints RDYIMG
//   Host sends 224*224*3 uint8 RGB bytes in 64-byte chunks
//   MCU prints ACK after each chunk
//   MCU runs TFLite Micro inference
//   MCU prints PRED <class> <seconds> <score> class_<class>
//   MCU prints READY
//
// Required local files in src/:
//   main.cpp
//   model_data.h
//   noodle_serial.h
//   noodle_serial.cpp
//
// model_data.h should expose:
//   const unsigned int squeezenet1_1_tflite_len;
//   alignas(16) const unsigned char squeezenet1_1_tflite[];
// ============================================================

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <math.h>

#include "noodle_serial.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Some TFLM Arduino ports need this include for op registrations.
// If your installed TFLM version complains, adjust according to that library.
#include "tensorflow/lite/micro/kernels/micro_ops.h"

// ============================================================
// Configuration
// ============================================================

static constexpr uint16_t IMG_W = 224;
static constexpr uint16_t IMG_H = 224;
static constexpr uint16_t IMG_C = 3;
static constexpr size_t RX_BYTES = (size_t)IMG_W * IMG_H * IMG_C;

// NoodleBuffer peak for the same model was 7,047,488 bytes.
// TFLM may require a larger tensor arena due to planning, alignment,
// persistent tensors, temporary buffers, and operator scratch.
//
// For N16R8, start lower if allocation fails:
//   7UL * 1024UL * 1024UL
//
// For N32R16, you can try:
//   9UL * 1024UL * 1024UL
static constexpr size_t TENSOR_ARENA_SIZE = 7500UL * 1024UL;

// Use longer timeout than the default 1000 ms when waiting for the host.
static constexpr uint32_t IMG_HEADER_TIMEOUT_MS = 10000;
static constexpr uint32_t RX_TIMEOUT_MS = 10000;

// ============================================================
// Globals
// ============================================================

static uint8_t *rx_img = nullptr;
static uint8_t *tensor_arena = nullptr;

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;

// Increase if AllocateTensors reports missing op registrations.
static tflite::MicroMutableOpResolver<16> resolver;

// ============================================================
// Memory helpers
// ============================================================

static float frag_percent(size_t free_bytes, size_t largest_bytes) {
  if (free_bytes == 0) return 0.0f;
  return 100.0f * (1.0f - ((float)largest_bytes / (float)free_bytes));
}

static void print_mem(const char *tag) {
  const size_t heap_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  const size_t heap_largest = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
  const size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  const size_t psram_largest = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);

  Serial.printf("DBG_MEM %s heap=%u heap_largest=%u heap_frag=%.2f%% "
                "psram=%u psram_largest=%u psram_frag=%.2f%%\n",
                tag,
                (unsigned)heap_free,
                (unsigned)heap_largest,
                frag_percent(heap_free, heap_largest),
                (unsigned)psram_free,
                (unsigned)psram_largest,
                frag_percent(psram_free, psram_largest));
}

static void print_tensor_info(const char *name, const TfLiteTensor *t) {
  if (!t) {
    Serial.printf("TENSOR %s null\n", name);
    return;
  }

  Serial.printf("TENSOR %s type=%d bytes=%u dims=",
                name, (int)t->type, (unsigned)t->bytes);

  if (t->dims) {
    Serial.print("[");
    for (int i = 0; i < t->dims->size; ++i) {
      if (i) Serial.print(",");
      Serial.print(t->dims->data[i]);
    }
    Serial.print("]");
  } else {
    Serial.print("[]");
  }

  Serial.printf(" scale=%f zp=%d\n", t->params.scale, t->params.zero_point);
}

// ============================================================
// Input preprocessing
// ============================================================

static bool fill_input_tensor_from_rgb_u8(const uint8_t *rgb) {
  if (!input || !rgb) return false;

  if (input->type == kTfLiteFloat32) {
    float *x = input->data.f;
    if (!x) return false;

    // The .tflite graph already performs:
    //   (x - mean) / std
    // using SUB and MUL operators.
    // Therefore we only scale uint8 RGB to [0, 1].
    for (size_t i = 0; i < (size_t)IMG_W * IMG_H * IMG_C; ++i) {
      x[i] = (float)rgb[i] / 255.0f;
    }

    return true;
  }

  Serial.printf("ERR_INPUT_TYPE type=%d\n", (int)input->type);
  return false;
}

// ============================================================
// Output parsing
// ============================================================

static bool get_top1(int *best_idx, float *best_score) {
  if (!output || !best_idx || !best_score) return false;

  int n = 1;
  if (output->dims) {
    n = 1;
    for (int i = 0; i < output->dims->size; ++i) {
      n *= output->dims->data[i];
    }
  }

  if (n <= 0) return false;

  int best = 0;
  float bestv = -1.0e30f;

  if (output->type == kTfLiteFloat32) {
    const float *y = output->data.f;
    for (int i = 0; i < n; ++i) {
      if (y[i] > bestv) {
        bestv = y[i];
        best = i;
      }
    }
  } else if (output->type == kTfLiteUInt8) {
    const uint8_t *y = output->data.uint8;
    const float scale = output->params.scale;
    const int zp = output->params.zero_point;
    for (int i = 0; i < n; ++i) {
      const float v = ((int)y[i] - zp) * scale;
      if (v > bestv) {
        bestv = v;
        best = i;
      }
    }
  } else if (output->type == kTfLiteInt8) {
    const int8_t *y = output->data.int8;
    const float scale = output->params.scale;
    const int zp = output->params.zero_point;
    for (int i = 0; i < n; ++i) {
      const float v = ((int)y[i] - zp) * scale;
      if (v > bestv) {
        bestv = v;
        best = i;
      }
    }
  } else {
    Serial.printf("ERR_OUTPUT_TYPE type=%d\n", (int)output->type);
    return false;
  }

  *best_idx = best;
  *best_score = bestv;
  return true;
}

// ============================================================
// TFLM setup
// ============================================================

static bool setup_tflm() {
  print_mem("before_tflm");

  model = tflite::GetModel(squeezenet1_1_tflite);
  if (!model) {
    Serial.println("ERR_MODEL_NULL");
    return false;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERR_SCHEMA model=%d expected=%d\n",
                  (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
    return false;
  }

// Register operators expected by a SqueezeNet-like float32 graph.
resolver.AddConv2D();
resolver.AddMaxPool2D();
resolver.AddAveragePool2D();
resolver.AddConcatenation();
resolver.AddReshape();
resolver.AddSoftmax();
resolver.AddRelu();
resolver.AddPad();
resolver.AddMean();
resolver.AddFullyConnected();
resolver.AddDepthwiseConv2D();

// Extra elementwise ops often present in converted float models.
resolver.AddAdd();
resolver.AddSub();
resolver.AddMul();
resolver.AddDiv();

  tensor_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE,
                                             MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) {
    Serial.printf("ERR_ARENA_ALLOC size=%u\n", (unsigned)TENSOR_ARENA_SIZE);
    print_mem("arena_alloc_fail");
    return false;
  }

  Serial.printf("TFLM_ARENA size=%u\n", (unsigned)TENSOR_ARENA_SIZE);
  print_mem("after_arena_alloc");

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERR_ALLOCATE_TENSORS");
    print_mem("allocate_tensors_fail");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  print_tensor_info("input", input);
  print_tensor_info("output", output);

  Serial.printf("TFLM_ARENA_USED %u\n", (unsigned)interpreter->arena_used_bytes());
  print_mem("after_allocate_tensors");

  return true;
}

// ============================================================
// Arduino setup / loop
// ============================================================

void setup() {
  NoodleSerial::begin(NoodleSerial::BAUD);

  Serial.println();
  Serial.println("BOOT TFLite Micro SqueezeNet-1.1 224x224");
  Serial.println("Input: RGB 224x224x3 HWC uint8");
  Serial.println("Runtime: TFLite Micro tensor arena in PSRAM");
  Serial.printf("Model bytes: %u\n", (unsigned)squeezenet1_1_tflite_len);

  print_mem("boot");

  rx_img = (uint8_t *)heap_caps_malloc(RX_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!rx_img) {
    Serial.printf("ERR_RX_ALLOC bytes=%u\n", (unsigned)RX_BYTES);
    print_mem("rx_alloc_fail");
    while (true) delay(1000);
  }

  print_mem("after_rx_alloc");

  if (!setup_tflm()) {
    Serial.println("ERR_TFLM_SETUP");
    while (true) delay(1000);
  }

  // Clear any garbage or stale host bytes before announcing readiness.
  NoodleSerial::clear_input();
  NoodleSerial::print_ready();
}

void loop() {
  // Wait for a real IMG header before doing anything.
  // This prevents accidental inference at boot.
  if (!NoodleSerial::wait_for_img_header(IMG_HEADER_TIMEOUT_MS)) {
    // Keep the host synchronized without invoking the model.
    NoodleSerial::print_ready();
    return;
  }

  if (!NoodleSerial::recv_image_chunked(rx_img, RX_BYTES, RX_TIMEOUT_MS)) {
    NoodleSerial::print_ready();
    return;
  }

  print_mem("after_recv");

  if (!fill_input_tensor_from_rgb_u8(rx_img)) {
    Serial.println("ERR_FILL_INPUT");
    NoodleSerial::print_ready();
    return;
  }

  print_mem("after_input_fill");

  const int64_t t0 = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter->Invoke();
  const int64_t t1 = esp_timer_get_time();

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERR_INVOKE");
    print_mem("invoke_fail");
    NoodleSerial::print_ready();
    return;
  }

  print_mem("after_invoke");

  int pred = -1;
  float score = 0.0f;
  if (!get_top1(&pred, &score)) {
    Serial.println("ERR_TOP1");
    NoodleSerial::print_ready();
    return;
  }

  const float seconds = (float)(t1 - t0) / 1000000.0f;
  Serial.printf("PRED %d %.6f %.6f class_%d\n", pred, seconds, score, pred);

  print_mem("after_predict");
  NoodleSerial::print_ready();
}
