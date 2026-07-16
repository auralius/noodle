#include <Arduino.h>
#include <math.h>
#include <unistd.h>

#ifndef NOODLE_USE_NONE
#define NOODLE_USE_NONE
#endif

#ifndef NOODLE_POOL_MODE
#define NOODLE_POOL_MODE NOODLE_POOL_NONE
#endif

#include "noodle.h"
#include "noodle_serial.h"
#include "model_weights_tinyresnet.h"

// ============================================================
// RP2350 TinyResNet CIFAR-10 serial demo
// ============================================================
//
// Host protocol:
//   host -> "IMG"
//   MCU  -> "RDYIMG"
//   host -> 3072 RGB uint8 bytes in 64-byte chunks
//   MCU  -> "ACK" after every chunk
//   MCU  -> "PRED <id> <seconds> <confidence> <class_name>"
//   MCU  -> "READY"
//
// Host image layout:
//   HWC RGB uint8, 32 x 32 x 3.
//
// Noodle tensor layout:
//   CHW float32.
//
// Model:
//   Stem: 3x3, 3 -> 16
//   Stage 1: two identity residual blocks, 16 channels
//   Stage 2: projection block + identity block, 32 channels
//   Stage 3: projection block + identity block, 64 channels
//   GAP -> Dense 64 -> 10 -> Softmax
//
// The Colab exporter folds BatchNorm into every convolution and
// generates model_weights_tinyresnet.h.
// ============================================================

static constexpr uint16_t IMG_W = 32;
static constexpr uint16_t IMG_H = 32;
static constexpr uint16_t IMG_C = 3;
static constexpr size_t IMG_PIXELS = (size_t)IMG_W * IMG_H;
static constexpr size_t IMG_BYTES = IMG_PIXELS * IMG_C;

// Print detailed per-stage growth only during the first inference.
// A compact final memory line is printed after every inference.
#ifndef TINYRESNET_BENCHMARK_STAGE_LOGS
#define TINYRESNET_BENCHMARK_STAGE_LOGS 1
#endif

static uint32_t inference_count = 0;

// Largest activation is 16 x 32 x 32 = 16384 floats.
static constexpr size_t MAX_ACT_FLOATS = 16u * 32u * 32u;
static uint8_t RX_BYTES[IMG_BYTES];

// X holds the normalized RGB input.
// A and B alternate as residual-block outputs.
// C stores the second-convolution output.
// In projection blocks, the temporary Conv1 buffer is reused for the
// projected shortcut after Conv2 has consumed the Conv1 result.
static NoodleBuffer X;
static NoodleBuffer A;
static NoodleBuffer B;
static NoodleBuffer C;

static const char *CLASS_NAMES[10] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Per-channel CIFAR-10 normalization values read from the trained
// Keras input_norm layer. Input bytes are first scaled to [0, 1],
// then standardized as:
//     x_norm = (x - input_mean[c]) * input_inv_std[c]
static constexpr float input_mean[3] = {
    0.4915495f,
    0.48212427f,
    0.44642738f
};

static constexpr float input_inv_std[3] = {
    4.0460887f,
    4.1038300f,
    3.8206637f
};


static Pool no_pool() {
    Pool p;
    p.M = 1;
    p.T = 1;
    return p;
}

static void make_conv(ConvMem &conv,
                      uint16_t kernel,
                      uint16_t padding,
                      uint16_t stride,
                      const float *weight,
                      const float *bias,
                      Activation activation) {
    conv.K = kernel;
    conv.P = padding;
    conv.S = stride;
    conv.OP = 0;
    conv.weight = weight;
    conv.bias = bias;
    conv.act = activation;
}

static void make_fcn(FCNMem &fcn,
                     const float *weight,
                     const float *bias,
                     Activation activation) {
    fcn.weight = weight;
    fcn.bias = bias;
    fcn.act = activation;
}

// RP2350/newlib contiguous stack-to-heap gap estimate.
// This is useful for tracking changes across inference, but it is not a
// largest-free-block measurement and therefore is not used to claim heap
// fragmentation.
static int32_t rp2350_free_ram_bytes() {
    char stack_top;
    void *heap_end = sbrk(0);

    if (heap_end == (void *)-1) {
        return -1;
    }

    return (int32_t)((intptr_t)&stack_top - (intptr_t)heap_end);
}

static void print_memory_state(const char *tag) {
    const size_t x_bytes = noodle_buffer_capacity_bytes(&X);
    const size_t a_bytes = noodle_buffer_capacity_bytes(&A);
    const size_t b_bytes = noodle_buffer_capacity_bytes(&B);
    const size_t c_bytes = noodle_buffer_capacity_bytes(&C);

    const size_t logical_bytes =
        x_bytes + a_bytes + b_bytes + c_bytes;

    const size_t arena_capacity =
        noodle_buffer_arena_capacity_bytes();
    const size_t arena_used =
        noodle_buffer_arena_used_bytes();
    const size_t arena_headroom =
        (arena_capacity >= arena_used)
            ? (arena_capacity - arena_used)
            : 0;

    Serial.print(F("MEM "));
    Serial.print(tag);
    Serial.print(F(" run="));
    Serial.print((unsigned long)inference_count);

    Serial.print(F(" X="));
    Serial.print((unsigned long)x_bytes);
    Serial.print(F(" A="));
    Serial.print((unsigned long)a_bytes);
    Serial.print(F(" B="));
    Serial.print((unsigned long)b_bytes);
    Serial.print(F(" C="));
    Serial.print((unsigned long)c_bytes);

    Serial.print(F(" logical="));
    Serial.print((unsigned long)logical_bytes);
    Serial.print(F(" arena_capacity="));
    Serial.print((unsigned long)arena_capacity);
    Serial.print(F(" arena_used="));
    Serial.print((unsigned long)arena_used);
    Serial.print(F(" arena_headroom="));
    Serial.print((unsigned long)arena_headroom);

    Serial.print(F(" free_ram="));
    Serial.println((long)rp2350_free_ram_bytes());
}

static inline void benchmark_stage(const char *tag) {
#if TINYRESNET_BENCHMARK_STAGE_LOGS
    if (inference_count == 1) {
        print_memory_state(tag);
    }
#else
    (void)tag;
#endif
}

static bool hwc_rgb_u8_to_normalized_chw(const uint8_t *src,
                                          NoodleBuffer *dst) {
    float *out = noodle_buffer_require(dst, IMG_PIXELS * IMG_C);
    if (!out) return false;

    for (uint16_t y = 0; y < IMG_H; ++y) {
        for (uint16_t x = 0; x < IMG_W; ++x) {
            const size_t hwc = ((size_t)y * IMG_W + x) * 3u;
            const size_t pos = (size_t)y * IMG_W + x;

            for (uint16_t c = 0; c < IMG_C; ++c) {
                const float v = (float)src[hwc + c] * (1.0f / 255.0f);
                out[(size_t)c * IMG_PIXELS + pos] =
                    (v - input_mean[c]) * input_inv_std[c];
            }
        }
    }

    return true;
}

static bool run_identity_block(NoodleBuffer *input,
                               NoodleBuffer *temp,
                               NoodleBuffer *main2,
                               uint16_t channels,
                               uint16_t width,
                               const ConvMem &conv1,
                               const ConvMem &conv2) {
    const Pool p = no_pool();

    uint16_t out_w = noodle_conv_float(
        input, channels, channels, temp, width, conv1, p, nullptr);
    if (out_w != width) {
        Serial.print(F("ERR identity conv1 W="));
        Serial.println(out_w);
        return false;
    }

    out_w = noodle_conv_float(
        temp, channels, channels, main2, width, conv2, p, nullptr);
    if (out_w != width) {
        Serial.print(F("ERR identity conv2 W="));
        Serial.println(out_w);
        return false;
    }

    // temp is no longer needed after conv2, so it becomes the block output.
    const uint32_t count = (uint32_t)channels * width * width;
    if (noodle_add(input, main2, temp, count, ACT_RELU) != count) {
        Serial.println(F("ERR identity add"));
        return false;
    }

    return true;
}

static bool run_projection_block(NoodleBuffer *input,
                                 NoodleBuffer *temp,
                                 NoodleBuffer *main2,
                                 uint16_t in_channels,
                                 uint16_t out_channels,
                                 uint16_t in_width,
                                 const ConvMem &conv1,
                                 const ConvMem &conv2,
                                 const ConvMem &proj,
                                 uint16_t &out_width) {
    const Pool p = no_pool();

    uint16_t w_main = noodle_conv_float(
        input, in_channels, out_channels, temp, in_width, conv1, p, nullptr);
    if (w_main == 0) {
        Serial.println(F("ERR projection conv1"));
        return false;
    }

    uint16_t w_main2 = noodle_conv_float(
        temp, out_channels, out_channels, main2, w_main, conv2, p, nullptr);
    if (w_main2 != w_main) {
        Serial.print(F("ERR projection conv2 W="));
        Serial.println(w_main2);
        return false;
    }

    // Conv2 has already consumed the Conv1 result in temp, so that tensor is
    // dead. Reuse temp for the projected shortcut.
    uint16_t w_skip = noodle_conv_float(
        input, in_channels, out_channels, temp, in_width, proj, p, nullptr);
    if (w_skip != w_main) {
        Serial.print(F("ERR projection skip W="));
        Serial.println(w_skip);
        return false;
    }

    // In-place on temp:
    //     temp = ReLU(main2 + temp)
    const uint32_t count = (uint32_t)out_channels * w_main * w_main;
    if (noodle_add(main2, temp, temp, count, ACT_RELU) != count) {
        Serial.println(F("ERR projection add"));
        return false;
    }

    out_width = w_main;
    return true;
}

static bool predict() {
    const Pool p = no_pool();

    ConvMem stem;
    make_conv(stem, 3, 65535, 1, w01, b01, ACT_RELU);

    ConvMem s1b1c1, s1b1c2, s1b2c1, s1b2c2;
    make_conv(s1b1c1, 3, 65535, 1, w02, b02, ACT_RELU);
    make_conv(s1b1c2, 3, 65535, 1, w03, b03, ACT_NONE);
    make_conv(s1b2c1, 3, 65535, 1, w04, b04, ACT_RELU);
    make_conv(s1b2c2, 3, 65535, 1, w05, b05, ACT_NONE);

    ConvMem s2b1c1, s2b1c2, s2b1proj, s2b2c1, s2b2c2;
    make_conv(s2b1c1, 3, 65535, 2, w06, b06, ACT_RELU);
    make_conv(s2b1c2, 3, 65535, 1, w07, b07, ACT_NONE);
    make_conv(s2b1proj, 1, 0, 2, w08, b08, ACT_NONE);
    make_conv(s2b2c1, 3, 65535, 1, w09, b09, ACT_RELU);
    make_conv(s2b2c2, 3, 65535, 1, w10, b10, ACT_NONE);

    ConvMem s3b1c1, s3b1c2, s3b1proj, s3b2c1, s3b2c2;
    make_conv(s3b1c1, 3, 65535, 2, w11, b11, ACT_RELU);
    make_conv(s3b1c2, 3, 65535, 1, w12, b12, ACT_NONE);
    make_conv(s3b1proj, 1, 0, 2, w13, b13, ACT_NONE);
    make_conv(s3b2c1, 3, 65535, 1, w14, b14, ACT_RELU);
    make_conv(s3b2c2, 3, 65535, 1, w15, b15, ACT_NONE);

    FCNMem classifier;
    make_fcn(classifier, w16, b16, ACT_SOFTMAX);

    uint16_t width = 32;
    uint16_t channels = 3;
    const uint32_t start_us = micros();

    // Stem: X -> A, 3x32x32 -> 16x32x32.
    width = noodle_conv_float(
        &X, channels, 16, &A, width, stem, p, nullptr);
    if (width != 32) {
        Serial.print(F("ERR stem W="));
        Serial.println(width);
        return false;
    }
    channels = 16;
    benchmark_stage("after_stem");

    // Stage 1 block 1: A -> B.
    if (!run_identity_block(
            &A, &B, &C, channels, width, s1b1c1, s1b1c2)) {
        return false;
    }
    benchmark_stage("after_s1b1");

    // Stage 1 block 2: B -> A.
    if (!run_identity_block(
            &B, &A, &C, channels, width, s1b2c1, s1b2c2)) {
        return false;
    }
    benchmark_stage("after_s1b2");

    // Stage 2 projection block: A -> B, 16x32x32 -> 32x16x16.
    uint16_t next_width = 0;
    if (!run_projection_block(
            &A, &B, &C,
            16, 32, width,
            s2b1c1, s2b1c2, s2b1proj,
            next_width)) {
        return false;
    }
    channels = 32;
    width = next_width;
    benchmark_stage("after_s2proj");

    // Stage 2 identity block: B -> A.
    if (!run_identity_block(
            &B, &A, &C, channels, width, s2b2c1, s2b2c2)) {
        return false;
    }
    benchmark_stage("after_s2id");

    // Stage 3 projection block: A -> B, 32x16x16 -> 64x8x8.
    if (!run_projection_block(
            &A, &B, &C,
            32, 64, width,
            s3b1c1, s3b1c2, s3b1proj,
            next_width)) {
        return false;
    }
    channels = 64;
    width = next_width;
    benchmark_stage("after_s3proj");

    // Stage 3 identity block: B -> A.
    if (!run_identity_block(
            &B, &A, &C, channels, width, s3b2c1, s3b2c2)) {
        return false;
    }
    benchmark_stage("after_s3id");

    // GAP in-place: A[64 x 8 x 8] -> A[64].
    uint16_t vector_size = noodle_gap(&A, channels, width);
    if (vector_size != channels) {
        Serial.print(F("ERR GAP V="));
        Serial.println(vector_size);
        return false;
    }
    benchmark_stage("after_gap");

    // Dense + softmax: A[64] -> B[10].
    vector_size = noodle_fcn(
        &A, vector_size, 10, &B, classifier, nullptr);
    if (vector_size != 10) {
        Serial.print(F("ERR FC V="));
        Serial.println(vector_size);
        return false;
    }
    benchmark_stage("after_dense");

    uint16_t predicted = 0;
    float confidence = 0.0f;
    noodle_find_max(&B, 10, confidence, predicted);

    const float elapsed_s = (float)(micros() - start_us) * 1.0e-6f;

    char line[112];
    snprintf(line, sizeof(line),
             "PRED %u %.6f %.6f %s",
             predicted,
             elapsed_s,
             confidence,
             CLASS_NAMES[predicted]);
    Serial.println(line);
    Serial.flush();

    return true;
}

void setup() {
    NoodleSerial::begin(921600);
    NoodleSerial::clear_input();

    noodle_buffer_init(&X);
    noodle_buffer_init(&A);
    noodle_buffer_init(&B);
    noodle_buffer_init(&C);

    Serial.println(F("BOOT RP2350 TinyResNet CIFAR-10"));
    Serial.println(F("Input: RGB 32x32x3 HWC uint8"));
    Serial.println(F("Runtime: NoodleBuffer residual inference"));
    Serial.println(F("Buffers: X + A + B + C (projection reuses temp)"));

    Serial.println(F("Benchmark: first-run stage growth + final state"));
    print_memory_state("boot");

    NoodleSerial::print_ready();
}

void loop() {
    if (!NoodleSerial::wait_for_img_header()) {
        NoodleSerial::print_ready();
        return;
    }

    if (!NoodleSerial::recv_image_chunked(RX_BYTES, IMG_BYTES)) {
        NoodleSerial::print_ready();
        return;
    }

    if (!hwc_rgb_u8_to_normalized_chw(RX_BYTES, &X)) {
        Serial.println(F("ERR input conversion"));
        NoodleSerial::print_ready();
        return;
    }

    ++inference_count;
    benchmark_stage("after_input");

    if (predict()) {
        print_memory_state("after_predict");
    }

    NoodleSerial::print_ready();
}
