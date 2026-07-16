#include <Arduino.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tinyresnet_model_data.h"
#include "noodle_serial.h"

extern "C" void DebugLog(const char *s) {
    if (s) {
        Serial.print(F("[TFLM] "));
        Serial.println(s);
    }
}

// ============================================================
// RP2350 TensorFlow Lite Micro TinyResNet CIFAR-10 benchmark
// ============================================================
//
// Host protocol, intentionally identical to the Noodle firmware:
//
//   host -> "IMG"
//   MCU  -> "RDYIMG"
//   host -> 3072 RGB uint8 bytes in 64-byte chunks
//   MCU  -> "ACK" after every chunk
//   MCU  -> "PRED <id> <seconds> <confidence> <class_name>"
//   MCU  -> "READY"
//
// Model input:
//   float32 NHWC [1, 32, 32, 3], values in [0, 1]
//
// Model output:
//   float32 [1, 10]
//
// The sender still transmits the original HWC RGB uint8 bytes.
// Firmware converts uint8 -> float32 [0, 1]. The trained normalization
// layer remains inside the TFLite graph.
// ============================================================


// ------------------------------------------------------------
// Serial protocol configuration
// ------------------------------------------------------------

static constexpr uint32_t SERIAL_BAUD = 921600;
static constexpr uint32_t SERIAL_TIMEOUT_MS = 5000;

static constexpr uint16_t IMG_W = 32;
static constexpr uint16_t IMG_H = 32;
static constexpr uint16_t IMG_C = 3;

static constexpr size_t IMG_BYTES =
    (size_t)IMG_W * IMG_H * IMG_C;

static constexpr size_t CHUNK_SIZE = 64;

// The host still sends raw RGB uint8 bytes. For the float32 model,
// receive them here first, then convert them to [0, 1] float32.
static uint8_t RX_BYTES[IMG_BYTES];


// ------------------------------------------------------------
// TFLM tensor arena
//
// Start generously. After AllocateTensors(), the firmware prints
// arena_used_bytes(), which can then be used to reduce this value.
// ------------------------------------------------------------

#ifndef TFLM_TENSOR_ARENA_BYTES
#define TFLM_TENSOR_ARENA_BYTES (320u * 1024u)
#endif

alignas(16) static uint8_t tensor_arena[TFLM_TENSOR_ARENA_BYTES];


// ------------------------------------------------------------
// Model and interpreter objects
// ------------------------------------------------------------

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

// Operators reported by the exported uint8 model:
// ADD, CONV_2D, FULLY_CONNECTED, MEAN, MUL, QUANTIZE,
// SOFTMAX, SUB.
//
// DELEGATE is a desktop-interpreter artifact and is not registered.
static tflite::MicroMutableOpResolver<8> resolver;

static uint32_t inference_count = 0;


// ------------------------------------------------------------
// CIFAR-10 labels
// ------------------------------------------------------------

static const char *CLASS_NAMES[10] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
};


// ------------------------------------------------------------
// RP2350 memory estimate
//
// This is the contiguous stack-to-heap gap estimate. It is useful
// for observing stability across repeated inference, but it is not
// a largest-free-block measurement and is therefore not presented
// as a heap-fragmentation metric.
// ------------------------------------------------------------

static int32_t rp2350_free_ram_bytes() {
    char stack_top;
    void *heap_end = sbrk(0);

    if (heap_end == (void *)-1) {
        return -1;
    }

    return (int32_t)(
        (intptr_t)&stack_top -
        (intptr_t)heap_end
    );
}


// ------------------------------------------------------------
// Small print helpers
// ------------------------------------------------------------

static const char *tensor_type_name(TfLiteType type) {
    switch (type) {
        case kTfLiteUInt8:
            return "uint8";
        case kTfLiteInt8:
            return "int8";
        case kTfLiteFloat32:
            return "float32";
        case kTfLiteInt32:
            return "int32";
        default:
            return "other";
    }
}


static void print_tensor_shape(
    const char *name,
    const TfLiteTensor *tensor
) {
    Serial.print(name);
    Serial.print(F(" shape=["));

    if (tensor && tensor->dims) {
        for (int i = 0; i < tensor->dims->size; ++i) {
            if (i) {
                Serial.print(',');
            }
            Serial.print(tensor->dims->data[i]);
        }
    }

    Serial.println(F("]"));
}


static void print_tensor_info(
    const char *name,
    const TfLiteTensor *tensor
) {
    if (!tensor) {
        Serial.print(name);
        Serial.println(F(": null"));
        return;
    }

    print_tensor_shape(name, tensor);

    Serial.print(name);
    Serial.print(F(" type="));
    Serial.print(tensor_type_name(tensor->type));
    Serial.print(F(" bytes="));
    Serial.print((unsigned long)tensor->bytes);
    Serial.print(F(" scale="));
    Serial.print(tensor->params.scale, 10);
    Serial.print(F(" zero_point="));
    Serial.println((long)tensor->params.zero_point);
}


static void print_memory_state(const char *tag) {
    const size_t arena_used =
        interpreter ? interpreter->arena_used_bytes() : 0;

    const size_t arena_headroom =
        (TFLM_TENSOR_ARENA_BYTES >= arena_used)
            ? (TFLM_TENSOR_ARENA_BYTES - arena_used)
            : 0;

    Serial.print(F("MEM "));
    Serial.print(tag);

    Serial.print(F(" run="));
    Serial.print((unsigned long)inference_count);

    Serial.print(F(" model="));
    Serial.print((unsigned long)g_tinyresnet_float32_model_data_len);

    Serial.print(F(" arena_capacity="));
    Serial.print((unsigned long)TFLM_TENSOR_ARENA_BYTES);

    Serial.print(F(" arena_used="));
    Serial.print((unsigned long)arena_used);

    Serial.print(F(" arena_headroom="));
    Serial.print((unsigned long)arena_headroom);

    Serial.print(F(" input="));
    Serial.print(
        input_tensor
            ? (unsigned long)input_tensor->bytes
            : 0ul
    );

    Serial.print(F(" output="));
    Serial.print(
        output_tensor
            ? (unsigned long)output_tensor->bytes
            : 0ul
    );

    Serial.print(F(" free_ram="));
    Serial.println((long)rp2350_free_ram_bytes());
}


// ------------------------------------------------------------
// Serial protocol
// ------------------------------------------------------------
//
// Reuse the exact same NoodleSerial implementation as the Noodle
// benchmark so host timing, READY behavior, chunk acknowledgements,
// and recovery behavior remain directly comparable.

// ------------------------------------------------------------
// TFLM setup
// ------------------------------------------------------------

static bool register_operators() {
    if (resolver.AddAdd() != kTfLiteOk) {
        Serial.println(F("ERR register ADD"));
        return false;
    }

    if (resolver.AddConv2D() != kTfLiteOk) {
        Serial.println(F("ERR register CONV_2D"));
        return false;
    }

    if (resolver.AddFullyConnected() != kTfLiteOk) {
        Serial.println(F("ERR register FULLY_CONNECTED"));
        return false;
    }

    if (resolver.AddMean() != kTfLiteOk) {
        Serial.println(F("ERR register MEAN"));
        return false;
    }

    if (resolver.AddMul() != kTfLiteOk) {
        Serial.println(F("ERR register MUL"));
        return false;
    }

    if (resolver.AddQuantize() != kTfLiteOk) {
        Serial.println(F("ERR register QUANTIZE"));
        return false;
    }

    if (resolver.AddSoftmax() != kTfLiteOk) {
        Serial.println(F("ERR register SOFTMAX"));
        return false;
    }

    if (resolver.AddSub() != kTfLiteOk) {
        Serial.println(F("ERR register SUB"));
        return false;
    }

    return true;
}


static bool setup_tflm() {
    model = tflite::GetModel(g_tinyresnet_float32_model_data);

    if (!model) {
        Serial.println(F("ERR GetModel"));
        return false;
    }

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print(F("ERR schema model="));
        Serial.print(model->version());
        Serial.print(F(" runtime="));
        Serial.println(TFLITE_SCHEMA_VERSION);
        return false;
    }

    if (!register_operators()) {
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        TFLM_TENSOR_ARENA_BYTES
    );

    interpreter = &static_interpreter;

    const TfLiteStatus allocation_status =
        interpreter->AllocateTensors();

    if (allocation_status != kTfLiteOk) {
        Serial.println(F("ERR AllocateTensors"));
        Serial.print(F("Try increasing TFLM_TENSOR_ARENA_BYTES from "));
        Serial.println((unsigned long)TFLM_TENSOR_ARENA_BYTES);
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    if (!input_tensor || !output_tensor) {
        Serial.println(F("ERR missing input/output tensor"));
        return false;
    }

    if (input_tensor->type != kTfLiteFloat32) {
        Serial.print(F("ERR input type="));
        Serial.println(tensor_type_name(input_tensor->type));
        return false;
    }

    if (output_tensor->type != kTfLiteFloat32) {
        Serial.print(F("ERR output type="));
        Serial.println(tensor_type_name(output_tensor->type));
        return false;
    }

    const size_t expected_input_bytes =
        IMG_BYTES * sizeof(float);

    if (input_tensor->bytes != expected_input_bytes) {
        Serial.print(F("ERR input bytes="));
        Serial.print((unsigned long)input_tensor->bytes);
        Serial.print(F(" expected="));
        Serial.println((unsigned long)expected_input_bytes);
        return false;
    }

    const size_t expected_output_bytes =
        10u * sizeof(float);

    if (output_tensor->bytes != expected_output_bytes) {
        Serial.print(F("ERR output bytes="));
        Serial.print((unsigned long)output_tensor->bytes);
        Serial.print(F(" expected="));
        Serial.println((unsigned long)expected_output_bytes);
        return false;
    }

    return true;
}


// ------------------------------------------------------------
// Inference
// ------------------------------------------------------------

static bool predict() {
    if (!interpreter || !output_tensor) {
        Serial.println(F("ERR interpreter not ready"));
        return false;
    }

    const uint32_t start_us = micros();

    const TfLiteStatus invoke_status =
        interpreter->Invoke();

    const uint32_t elapsed_us =
        (uint32_t)(micros() - start_us);

    if (invoke_status != kTfLiteOk) {
        Serial.println(F("ERR Invoke"));
        return false;
    }

    const float *scores = output_tensor->data.f;

    uint16_t predicted = 0;
    float confidence = scores[0];

    for (uint16_t i = 1; i < 10; ++i) {
        if (scores[i] > confidence) {
            confidence = scores[i];
            predicted = i;
        }
    }

    const float elapsed_s =
        (float)elapsed_us * 1.0e-6f;

    char line[112];

    snprintf(
        line,
        sizeof(line),
        "PRED %u %.6f %.6f %s",
        predicted,
        elapsed_s,
        confidence,
        CLASS_NAMES[predicted]
    );

    Serial.println(line);
    Serial.flush();

    return true;
}


// ------------------------------------------------------------
// Arduino entry points
// ------------------------------------------------------------

void setup() {
    NoodleSerial::begin(SERIAL_BAUD);
    NoodleSerial::clear_input();

    Serial.println(F("BOOT RP2350 TFLM TinyResNet CIFAR-10"));
    Serial.println(F("Input: RGB 32x32x3 HWC uint8"));
    Serial.println(F("Runtime: TensorFlow Lite Micro float32"));
    Serial.println(F("Protocol: IMG/RDYIMG/ACK/PRED/READY"));
    Serial.println(F("Operators: ADD CONV_2D FULLY_CONNECTED MEAN"));
    Serial.println(F("           MUL QUANTIZE SOFTMAX SUB"));

    Serial.print(F("Model bytes: "));
    Serial.println((unsigned long)g_tinyresnet_float32_model_data_len);

    Serial.print(F("Tensor arena capacity: "));
    Serial.println((unsigned long)TFLM_TENSOR_ARENA_BYTES);

    if (!setup_tflm()) {
        Serial.println(F("FATAL TFLM setup failed"));

        for (;;) {
            delay(1000);
        }
    }

    print_tensor_info("Input", input_tensor);
    print_tensor_info("Output", output_tensor);

    Serial.print(F("Tensor arena used: "));
    Serial.println(
        (unsigned long)interpreter->arena_used_bytes()
    );

    Serial.print(F("Tensor arena headroom: "));
    Serial.println(
        (unsigned long)(
            TFLM_TENSOR_ARENA_BYTES -
            interpreter->arena_used_bytes()
        )
    );

    print_memory_state("after_allocate");

    NoodleSerial::print_ready();
}


void loop() {
    if (!NoodleSerial::wait_for_img_header()) {
        NoodleSerial::print_ready();
        return;
    }

    if (!input_tensor ||
        input_tensor->type != kTfLiteFloat32 ||
        !input_tensor->data.f ||
        input_tensor->bytes != IMG_BYTES * sizeof(float)) {
        Serial.println(F("ERR invalid float32 TFLM input tensor"));
        NoodleSerial::print_ready();
        return;
    }

    if (!NoodleSerial::recv_image_chunked(
            RX_BYTES,
            IMG_BYTES)) {
        NoodleSerial::print_ready();
        return;
    }

    float *input = input_tensor->data.f;
    for (size_t i = 0; i < IMG_BYTES; ++i) {
        input[i] = (float)RX_BYTES[i] * (1.0f / 255.0f);
    }

    ++inference_count;

    if (inference_count == 1) {
        print_memory_state("before_first_invoke");
    }

    if (predict()) {
        print_memory_state("after_predict");
    }

    NoodleSerial::print_ready();
}
