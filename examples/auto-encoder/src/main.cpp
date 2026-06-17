// Deep Sequential AE-28 inference example for the Noodle framework.
// Last tested target: ESP32-S3 N18R8
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
//   Do not print debug text between image reception and OUT.
//   The Python sender's read_line() can desynchronize if text lines and
//   binary image output arrive in the same USB packet.

#include <Arduino.h>
#include <esp_heap_caps.h>

#include "noodle.h"
#include "noodle_serial.h"

#include "w01.h"
#include "b01.h"
#include "w02.h"
#include "b02.h"
#include "w03.h"
#include "b03.h"
#include "w04.h"
#include "b04.h"
#include "w05.h"
#include "b05.h"
#include "w06.h"
#include "b06.h"
#include "w07.h"
#include "b07.h"
#include "w08.h"
#include "b08.h"
#include "w09.h"
#include "b09.h"
#include "w10.h"
#include "b10.h"
#include "w11.h"
#include "b11.h"

static constexpr uint16_t IMG_W = 28;
static constexpr uint16_t IMG_SIZE = IMG_W * IMG_W;

// Largest activation in Deep Sequenntial AE-28:
//   28 x 28 x 16 = 12544 floats = 50176 bytes.
//
// With two grow-only ping-pong tensors, expected visible A+B capacity:
//   2 x 12544 floats = 25088 floats = 100352 bytes.
static constexpr uint32_t PEAK_FLOATS = 28UL * 28UL * 16UL;

static uint8_t RX_BYTES[IMG_SIZE];
static uint8_t TX_BYTES[IMG_SIZE];

static NoodleTensor A;
static NoodleTensor B;
static NoodleTensor *Y = &A;

static inline void swap_tensors(NoodleTensor *&x, NoodleTensor *&y) {
  NoodleTensor *t = x;
  x = y;
  y = t;
}

static float fragmentation_percent(size_t free_bytes, size_t largest_block) {
  if (free_bytes == 0) {
    return 0.0f;
  }

  float frag = 100.0f * (1.0f - ((float)largest_block / (float)free_bytes));

  if (frag < 0.0f) {
    frag = 0.0f;
  }
  if (frag > 100.0f) {
    frag = 100.0f;
  }

  return frag;
}

static void print_memory_state(const char *tag) {
  const size_t heap_free =
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  const size_t heap_largest =
      heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

  const size_t psram_free =
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  const size_t psram_largest =
      heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  const float heap_frag = fragmentation_percent(heap_free, heap_largest);
  const float psram_frag = fragmentation_percent(psram_free, psram_largest);

  Serial.printf(
      "MEM %s "
      "A_cap=%lu A_C=%u A_W=%u A_rank=%u "
      "B_cap=%lu B_C=%u B_W=%u B_rank=%u "
      "total=%lu bytes=%lu "
      "heap=%u heap_largest=%u heap_frag=%.2f%% "
      "psram=%u psram_largest=%u psram_frag=%.2f%%\n",
      tag,
      (unsigned long)A.buffer.capacity,
      A.C,
      A.W,
      A.rank,
      (unsigned long)B.buffer.capacity,
      B.C,
      B.W,
      B.rank,
      (unsigned long)(A.buffer.capacity + B.buffer.capacity),
      (unsigned long)((A.buffer.capacity + B.buffer.capacity) * sizeof(float)),
      (unsigned int)heap_free,
      (unsigned int)heap_largest,
      heap_frag,
      (unsigned int)psram_free,
      (unsigned int)psram_largest,
      psram_frag);
}

static void bytes_to_float_image(const uint8_t *src, float *dst, size_t n) {
  const float inv = 1.0f / 255.0f;
  for (size_t i = 0; i < n; i++) {
    dst[i] = (float)src[i] * inv;
  }
}

static void float_image_to_bytes(const float *src, uint8_t *dst, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float v = src[i];

    if (v < 0.0f) {
      v = 0.0f;
    }
    if (v > 1.0f) {
      v = 1.0f;
    }

    dst[i] = (uint8_t)(255.0f * v + 0.5f);
  }
}

static uint16_t run_autoencoder_forward() {
  // Encoder
  ConvMem L01;
  L01.K = 3; L01.P = 65535; L01.S = 1; L01.OP = 0; L01.O = 16;
  L01.weight = w01; L01.bias = b01; L01.act = ACT_RELU;

  ConvMem L02;
  L02.K = 3; L02.P = 65535; L02.S = 1; L02.OP = 0; L02.O = 16;
  L02.weight = w02; L02.bias = b02; L02.act = ACT_RELU;

  ConvMem L03;
  L03.K = 3; L03.P = 65535; L03.S = 2; L03.OP = 0; L03.O = 32;
  L03.weight = w03; L03.bias = b03; L03.act = ACT_RELU;

  ConvMem L04;
  L04.K = 3; L04.P = 65535; L04.S = 1; L04.OP = 0; L04.O = 32;
  L04.weight = w04; L04.bias = b04; L04.act = ACT_RELU;

  ConvMem L05;
  L05.K = 3; L05.P = 65535; L05.S = 2; L05.OP = 0; L05.O = 64;
  L05.weight = w05; L05.bias = b05; L05.act = ACT_RELU;

  ConvMem L06;
  L06.K = 3; L06.P = 65535; L06.S = 1; L06.OP = 0; L06.O = 64;
  L06.weight = w06; L06.bias = b06; L06.act = ACT_RELU;

  // Decoder
  ConvMem L07;
  L07.K = 3; L07.P = 65535; L07.S = 2; L07.OP = 1; L07.O = 32;
  L07.weight = w07; L07.bias = b07; L07.act = ACT_RELU;

  ConvMem L08;
  L08.K = 3; L08.P = 65535; L08.S = 1; L08.OP = 0; L08.O = 32;
  L08.weight = w08; L08.bias = b08; L08.act = ACT_RELU;

  ConvMem L09;
  L09.K = 3; L09.P = 65535; L09.S = 2; L09.OP = 1; L09.O = 16;
  L09.weight = w09; L09.bias = b09; L09.act = ACT_RELU;

  ConvMem L10;
  L10.K = 3; L10.P = 65535; L10.S = 1; L10.OP = 0; L10.O = 16;
  L10.weight = w10; L10.bias = b10; L10.act = ACT_RELU;

  ConvMem L11;
  L11.K = 3; L11.P = 65535; L11.S = 1; L11.OP = 0; L11.O = 1;
  L11.weight = w11; L11.bias = b11; L11.act = ACT_NONE;

  Pool no_pool;
  no_pool.M = 1;
  no_pool.T = 1;

  NoodleTensor *in = &A;
  NoodleTensor *out = &B;

  // Input should already be shaped as:
  //   A = 28 x 28 x 1
  if (in->rank != 2 || in->C != 1 || in->W != IMG_W || !in->buffer.data) {
    return 0;
  }

  if (!noodle_conv2d(in, out, L01, no_pool)) return 0;       // 28 x 28 x 16
  swap_tensors(in, out);

  if (!noodle_conv2d(in, out, L02, no_pool)) return 0;       // 28 x 28 x 16
  swap_tensors(in, out);

  if (!noodle_conv2d(in, out, L03, no_pool)) return 0;       // 14 x 14 x 32
  swap_tensors(in, out);

  if (!noodle_conv2d(in, out, L04, no_pool)) return 0;       // 14 x 14 x 32
  swap_tensors(in, out);

  if (!noodle_conv2d(in, out, L05, no_pool)) return 0;       // 7 x 7 x 64
  swap_tensors(in, out);

  if (!noodle_conv2d(in, out, L06, no_pool)) return 0;       // 7 x 7 x 64
  swap_tensors(in, out);

  if (!noodle_conv_transpose2d(in, out, L07)) return 0;      // 14 x 14 x 32
  swap_tensors(in, out);

  if (!noodle_conv2d(in, out, L08, no_pool)) return 0;       // 14 x 14 x 32
  swap_tensors(in, out);

  if (!noodle_conv_transpose2d(in, out, L09)) return 0;      // 28 x 28 x 16
  swap_tensors(in, out);

  if (!noodle_conv2d(in, out, L10, no_pool)) return 0;       // 28 x 28 x 16
  swap_tensors(in, out);

  if (!noodle_conv2d(in, out, L11, no_pool)) return 0;       // 28 x 28 x 1
  swap_tensors(in, out);

  if (in->rank != 2 || in->C != 1 || in->W != IMG_W || !in->buffer.data) {
    return 0;
  }

  noodle_sigmoid(&in->buffer, IMG_SIZE);

  Y = in;
  return in->W;
}

static void process_one_image() {
  float *x = noodle_tensor_require_2d(&A, 1, IMG_W);
  if (!x) {
    Serial.println(F("ERR_ALLOC_INPUT"));
    NoodleSerial::print_ready();
    return;
  }

  bytes_to_float_image(RX_BYTES, x, IMG_SIZE);

  const uint32_t t0 = micros();
  const uint16_t W = run_autoencoder_forward();
  const uint32_t dt = micros() - t0;

  if (W != IMG_W) {
    Serial.printf("ERR_BAD_W_FINAL %u\n", W);
    NoodleSerial::print_ready();
    return;
  }

  if (!Y || !Y->buffer.data || Y->rank != 2 || Y->C != 1 || Y->W != IMG_W) {
    Serial.printf("ERR_BAD_FINAL_SHAPE C=%u W=%u rank=%u\n",
                  Y ? Y->C : 0,
                  Y ? Y->W : 0,
                  Y ? Y->rank : 0);
    NoodleSerial::print_ready();
    return;
  }

  float_image_to_bytes(Y->buffer.data, TX_BYTES, IMG_SIZE);

  // No debug prints after this point.
  // The sender expects OUT line followed immediately by exactly 784 binary bytes.
  NoodleSerial::send_output_image(TX_BYTES, IMG_SIZE, dt);
}

static bool read_exact_command3(char cmd[4]) {
  cmd[0] = 0;
  cmd[1] = 0;
  cmd[2] = 0;
  cmd[3] = 0;

  // Non-blocking command read.
  // Only consume bytes when a complete 3-byte command is already available.
  if (Serial.available() < 3) {
    return false;
  }

  const size_t n = Serial.readBytes(cmd, 3);
  return n == 3;
}

static bool cmd_is(const char cmd[4], char a, char b, char c) {
  return cmd[0] == a && cmd[1] == b && cmd[2] == c;
}

void setup() {
  NoodleSerial::begin();
  NoodleSerial::clear_input();

  Serial.println();
  Serial.println(F("BOOT NOODLETENSOR DEEPSEQ_AE28 MEM"));

  noodle_tensor_init(&A);
  noodle_tensor_init(&B);

  print_memory_state("boot_before_prealloc");

  // Preallocate both ping-pong tensors to remove first-growth cost.
  //
  // Largest activation:
  //   28 x 28 x 16
  //
  // This sets the current metadata too, but that is fine because each image
  // call resets A to 1 x 28 x 28 using noodle_tensor_require_2d(&A, 1, IMG_W).
  float *a = noodle_tensor_require_2d(&A, 16, IMG_W);
  float *b = noodle_tensor_require_2d(&B, 16, IMG_W);

  if (!a || !b) {
    Serial.println(F("ERR_ALLOC_INIT"));
    while (true) {
      delay(1000);
    }
  }

  // Put A back into input shape for readability.
  // Capacity remains 28 x 28 x 16 because NoodleTensor is grow-only.
  if (!noodle_tensor_require_2d(&A, 1, IMG_W)) {
    Serial.println(F("ERR_ALLOC_INPUT_SHAPE"));
    while (true) {
      delay(1000);
    }
  }

  print_memory_state("after_prealloc");

  NoodleSerial::print_ready();
}

void loop() {
  // Command dispatcher.
  //
  // We avoid Serial.peekBytes() because USB CDC Serial on ESP32-S3 (HWCDC)
  // does not implement peekBytes(). Instead, we consume the 3-byte command.
  //
  // Supported commands:
  //   IMG : followed by chunked 784-byte image payload
  //   MEM : text-only memory query, safe between images
  char cmd[4];

  if (!read_exact_command3(cmd)) {
    // Stay quiet while idle.
    // Printing READY repeatedly can fill the PC-side serial buffer with stale
    // READY lines. Then the sender may spend a long time reading old READYs
    // after it has already sent IMG, and the protocol appears stuck.
    delay(5);
    return;
  }

  if (cmd_is(cmd, 'M', 'E', 'M')) {
    print_memory_state("query");
    NoodleSerial::print_ready();
    return;
  }

  if (cmd_is(cmd, 'I', 'M', 'G')) {
    // Header has already been consumed here, so we directly receive chunks.
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