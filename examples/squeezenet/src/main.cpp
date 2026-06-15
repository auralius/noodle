#include <Arduino.h>
#include <stdlib.h>

#ifndef NOODLE_USE_NONE
#define NOODLE_USE_NONE
#endif
#ifndef NOODLE_POOL_MODE
#define NOODLE_POOL_MODE NOODLE_POOL_MAX
#endif

#if defined(ARDUINO_ARCH_ESP32)
#include <esp_heap_caps.h>
#endif

#include "noodle.h"
#include "noodle_serial.h"
#include "model_weights.h"

// ============================================================
// Full SqueezeNet-1.1-style ImageNet serial demo for Noodle
// ============================================================
//
// Target experiment:
//   ESP32 / ESP32-S3 with PSRAM, or another board with enough RAM.
//
// Input protocol:
//   Host sends "IMG"
//   Device replies "RDYIMG"
//   Host sends 224*224*3 RGB bytes in chunks
//   Device replies "ACK" after each chunk
//   Device runs SqueezeNet-1.1-style inference
//   Device prints:
//       PRED <class_id> <seconds> <confidence> class_<class_id>
//   Device prints READY
//
// Input image layout from host:
//   HWC RGB uint8: [y][x][r,g,b], 150528 bytes
//
// Noodle tensor layout:
//   CHW float: [c][y][x], normalized to [0, 1]
//
// Model: SqueezeNet 1.1 style
//
//   Input RGB              3 x 224 x 224
//   Conv1 3x3/s2           64 x 111 x 111
//   MaxPool 3x3/s2         64 x 55 x 55
//
//   Fire2                  128 x 55 x 55
//   Fire3                  128 x 55 x 55
//   MaxPool 3x3/s2         128 x 27 x 27
//
//   Fire4                  256 x 27 x 27
//   Fire5                  256 x 27 x 27
//   MaxPool 3x3/s2         256 x 13 x 13
//
//   Fire6                  384 x 13 x 13
//   Fire7                  384 x 13 x 13
//   Fire8                  512 x 13 x 13
//   Fire9                  512 x 13 x 13
//
//   Final Conv1x1          1000 x 13 x 13
//   GAP                    1000
//   Softmax                1000
//
// Required model_weights.h:
//   This file must define w01..w26 and b01..b26.
//
// Layer mapping:
//   w01  conv1
//   w02  fire2 squeeze      w03  fire2 expand1x1     w04  fire2 expand3x3
//   w05  fire3 squeeze      w06  fire3 expand1x1     w07  fire3 expand3x3
//   w08  fire4 squeeze      w09  fire4 expand1x1     w10  fire4 expand3x3
//   w11  fire5 squeeze      w12  fire5 expand1x1     w13  fire5 expand3x3
//   w14  fire6 squeeze      w15  fire6 expand1x1     w16  fire6 expand3x3
//   w17  fire7 squeeze      w18  fire7 expand1x1     w19  fire7 expand3x3
//   w20  fire8 squeeze      w21  fire8 expand1x1     w22  fire8 expand3x3
//   w23  fire9 squeeze      w24  fire9 expand1x1     w25  fire9 expand3x3
//   w26  final classifier conv1x1, 512 -> 1000
//
// Important:
//   The current Tiny FireNet CIFAR-10 model_weights.h is NOT compatible.
//   You must export full SqueezeNet weights with this exact layer order.
// ============================================================

#define PRINT_MEM_DEBUG 1

static const uint16_t IMG_W = 224;
static const uint16_t IMG_H = 224;
static const uint16_t IMG_C = 3;
static const uint32_t IMG_PIXELS = (uint32_t)IMG_W * IMG_H;
static const uint32_t IMG_BYTES = IMG_PIXELS * IMG_C;
static const uint16_t NUM_CLASSES = 1000;

// Allocate input bytes dynamically. On ESP32, prefer PSRAM.
static uint8_t *RX_BYTES = nullptr;

// Main tensor buffers.
static NoodleBuffer X; // input tensor
static NoodleBuffer A; // main tensor buffer
static NoodleBuffer B; // main tensor buffer

// Fire-module scratch buffers.
// S must remain alive while expand1 and expand3 are computed.
// E1 and E3 must remain alive until noodle_concat() finishes.
static NoodleBuffer S;
static NoodleBuffer E1;
static NoodleBuffer E3;

// ============================================================
// Memory helpers
// ============================================================

static uint8_t *alloc_u8_buffer(size_t n)
{
#if defined(ARDUINO_ARCH_ESP32)
  if (psramFound())
  {
    uint8_t *p = (uint8_t *)ps_malloc(n);
    if (p)
      return p;
  }
#endif
  return (uint8_t *)malloc(n);
}

static int free_ram_approx()
{
#if defined(ARDUINO_ARCH_ESP32)
  return (int)ESP.getFreeHeap();
#else
  return -1;
#endif
}

static float fragmentation_percent(size_t free_bytes, size_t largest_block)
{
  if (free_bytes == 0)
    return 0.0f;

  float frag = 100.0f * (1.0f - ((float)largest_block / (float)free_bytes));

  // Clamp small numerical artifacts.
  if (frag < 0.0f)
    frag = 0.0f;
  if (frag > 100.0f)
    frag = 100.0f;

  return frag;
}

static void print_free_memory(const char *tag)
{
#if PRINT_MEM_DEBUG
#if defined(ARDUINO_ARCH_ESP32)

  const size_t heap_free =
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

  const size_t heap_largest =
      heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

  const size_t psram_free =
      psramFound() ? heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) : 0;

  const size_t psram_largest =
      psramFound() ? heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) : 0;

  const float heap_frag = fragmentation_percent(heap_free, heap_largest);
  const float psram_frag = fragmentation_percent(psram_free, psram_largest);

  char line[240];
  snprintf(line, sizeof(line),
           "DBG_MEM %s heap=%u heap_largest=%u heap_frag=%.2f%% "
           "psram=%u psram_largest=%u psram_frag=%.2f%%",
           tag,
           (unsigned)heap_free,
           (unsigned)heap_largest,
           heap_frag,
           (unsigned)psram_free,
           (unsigned)psram_largest,
           psram_frag);

  Serial.println(line);

#else
  char line[80];
  snprintf(line, sizeof(line), "DBG_MEM %s free=%d", tag, free_ram_approx());
  Serial.println(line);
#endif
  Serial.flush();
#endif
}

static void print_buffer_memory(const char *tag)
{
#if PRINT_MEM_DEBUG
  const unsigned x_bytes = (unsigned)noodle_buffer_capacity_bytes(&X);
  const unsigned a_bytes = (unsigned)noodle_buffer_capacity_bytes(&A);
  const unsigned b_bytes = (unsigned)noodle_buffer_capacity_bytes(&B);
  const unsigned s_bytes = (unsigned)noodle_buffer_capacity_bytes(&S);
  const unsigned e1_bytes = (unsigned)noodle_buffer_capacity_bytes(&E1);
  const unsigned e3_bytes = (unsigned)noodle_buffer_capacity_bytes(&E3);
  const unsigned total = x_bytes + a_bytes + b_bytes + s_bytes + e1_bytes + e3_bytes;

  char line[220];
  snprintf(line, sizeof(line),
           "DBG_BUF %s X=%u A=%u B=%u S=%u E1=%u E3=%u total=%u bytes",
           tag,
           x_bytes, a_bytes, b_bytes, s_bytes, e1_bytes, e3_bytes, total);
  Serial.println(line);
  Serial.flush();
#endif
}

// ============================================================
// Small helpers
// ============================================================

static void make_conv(ConvMem &c,
                      uint16_t K,
                      uint16_t P,
                      uint16_t S_stride,
                      const float *w,
                      const float *b,
                      Activation act)
{
  c.K = K;
  c.P = P;
  c.S = S_stride;
  c.OP = 0;
  c.weight = w;
  c.bias = b;
  c.act = act;
}

static Pool no_pool()
{
  Pool p;
  p.M = 1;
  p.T = 1;
  return p;
}

static bool hwc_rgb_u8_to_chw_float(const uint8_t *src, NoodleBuffer *dst)
{
  float *x = noodle_buffer_require(dst, (size_t)IMG_PIXELS * IMG_C);
  if (!x)
    return false;

  for (uint16_t y = 0; y < IMG_H; ++y)
  {
    for (uint16_t xpix = 0; xpix < IMG_W; ++xpix)
    {
      const uint32_t hwc = ((uint32_t)y * IMG_W + xpix) * 3;
      const uint32_t chw0 = (uint32_t)y * IMG_W + xpix;

      // input normalization using: x = (x - mean) / std, mean and std is from PyTorch's SqueezeNet 1.1 preprocessing
      x[0 * IMG_PIXELS + chw0] = ((float)src[hwc + 0] * (1.0f / 255.0f) - 0.485f) / 0.229f;
      x[1 * IMG_PIXELS + chw0] = ((float)src[hwc + 1] * (1.0f / 255.0f) - 0.456f) / 0.224f;
      x[2 * IMG_PIXELS + chw0] = ((float)src[hwc + 2] * (1.0f / 255.0f) - 0.406f) / 0.225f;
    }
  }

  return true;
}

static uint16_t run_fire(NoodleBuffer *input,
                         uint16_t C_in,
                         uint16_t W,
                         NoodleBuffer *output,
                         const ConvMem &squeeze,
                         uint16_t C_sq,
                         const ConvMem &expand1,
                         uint16_t C_e1,
                         const ConvMem &expand3,
                         uint16_t C_e3)
{
  const Pool p = no_pool();

  uint16_t V = noodle_conv_float(input, C_in, C_sq, &S, W, squeeze, p, NULL);
  if (V != W)
  {
    Serial.print(F("ERR fire squeeze W="));
    Serial.println(V);
    return 0;
  }

  V = noodle_conv_float(&S, C_sq, C_e1, &E1, W, expand1, p, NULL);
  if (V != W)
  {
    Serial.print(F("ERR fire expand1 W="));
    Serial.println(V);
    return 0;
  }

  V = noodle_conv_float(&S, C_sq, C_e3, &E3, W, expand3, p, NULL);
  if (V != W)
  {
    Serial.print(F("ERR fire expand3 W="));
    Serial.println(V);
    return 0;
  }

  const uint16_t C_out = noodle_concat(&E1, C_e1, &E3, C_e3, output, W);
  if (C_out != (uint16_t)(C_e1 + C_e3))
  {
    Serial.println(F("ERR fire concat"));
    return 0;
  }

  return C_out;
}

// ============================================================
// Prediction
// ============================================================

static bool predict()
{
  const Pool p = no_pool();

  // SqueezeNet 1.1 layer descriptors.
  ConvMem conv1;
  make_conv(conv1, 3, 0, 2, w01, b01, ACT_RELU);

  ConvMem f2s, f2e1, f2e3;
  make_conv(f2s, 1, 0, 1, w02, b02, ACT_RELU);
  make_conv(f2e1, 1, 0, 1, w03, b03, ACT_RELU);
  make_conv(f2e3, 3, 65535, 1, w04, b04, ACT_RELU);

  ConvMem f3s, f3e1, f3e3;
  make_conv(f3s, 1, 0, 1, w05, b05, ACT_RELU);
  make_conv(f3e1, 1, 0, 1, w06, b06, ACT_RELU);
  make_conv(f3e3, 3, 65535, 1, w07, b07, ACT_RELU);

  ConvMem f4s, f4e1, f4e3;
  make_conv(f4s, 1, 0, 1, w08, b08, ACT_RELU);
  make_conv(f4e1, 1, 0, 1, w09, b09, ACT_RELU);
  make_conv(f4e3, 3, 65535, 1, w10, b10, ACT_RELU);

  ConvMem f5s, f5e1, f5e3;
  make_conv(f5s, 1, 0, 1, w11, b11, ACT_RELU);
  make_conv(f5e1, 1, 0, 1, w12, b12, ACT_RELU);
  make_conv(f5e3, 3, 65535, 1, w13, b13, ACT_RELU);

  ConvMem f6s, f6e1, f6e3;
  make_conv(f6s, 1, 0, 1, w14, b14, ACT_RELU);
  make_conv(f6e1, 1, 0, 1, w15, b15, ACT_RELU);
  make_conv(f6e3, 3, 65535, 1, w16, b16, ACT_RELU);

  ConvMem f7s, f7e1, f7e3;
  make_conv(f7s, 1, 0, 1, w17, b17, ACT_RELU);
  make_conv(f7e1, 1, 0, 1, w18, b18, ACT_RELU);
  make_conv(f7e3, 3, 65535, 1, w19, b19, ACT_RELU);

  ConvMem f8s, f8e1, f8e3;
  make_conv(f8s, 1, 0, 1, w20, b20, ACT_RELU);
  make_conv(f8e1, 1, 0, 1, w21, b21, ACT_RELU);
  make_conv(f8e3, 3, 65535, 1, w22, b22, ACT_RELU);

  ConvMem f9s, f9e1, f9e3;
  make_conv(f9s, 1, 0, 1, w23, b23, ACT_RELU);
  make_conv(f9e1, 1, 0, 1, w24, b24, ACT_RELU);
  make_conv(f9e3, 3, 65535, 1, w25, b25, ACT_RELU);

  ConvMem final_conv;
  make_conv(final_conv, 1, 0, 1, w26, b26, ACT_RELU);

  uint16_t W = IMG_W;
  uint16_t C = IMG_C;

  const uint32_t t_all = micros();

  // Conv1: X -> A, 3x224x224 -> 64x111x111
  W = noodle_conv_float(&X, C, 64, &A, W, conv1, p, NULL);
  if (W != 111)
  {
    Serial.print(F("ERR conv1 W="));
    Serial.println(W);
    return false;
  }
  C = 64;
  print_buffer_memory("after_conv1");
  print_free_memory("after_conv1");

  // Pool1: A -> B, 64x111x111 -> 64x55x55
  W = noodle_pool2d(&A, C, W, &B, 3, 2);
  if (W != 55)
  {
    Serial.print(F("ERR pool1 W="));
    Serial.println(W);
    return false;
  }
  print_buffer_memory("after_pool1");

  // Fire2: B -> A, 64x55x55 -> 128x55x55
  C = run_fire(&B, C, W, &A, f2s, 16, f2e1, 64, f2e3, 64);
  if (C != 128)
    return false;
  print_buffer_memory("after_fire2");
  print_free_memory("after_fire2");

  // Fire3: A -> B, 128x55x55 -> 128x55x55
  C = run_fire(&A, C, W, &B, f3s, 16, f3e1, 64, f3e3, 64);
  if (C != 128)
    return false;
  print_buffer_memory("after_fire3");

  // Pool2: B -> A, 128x55x55 -> 128x27x27
  W = noodle_pool2d(&B, C, W, &A, 3, 2);
  if (W != 27)
  {
    Serial.print(F("ERR pool2 W="));
    Serial.println(W);
    return false;
  }
  print_buffer_memory("after_pool2");
  print_free_memory("after_pool2");

  // Fire4: A -> B, 128x27x27 -> 256x27x27
  C = run_fire(&A, C, W, &B, f4s, 32, f4e1, 128, f4e3, 128);
  if (C != 256)
    return false;
  print_buffer_memory("after_fire4");

  // Fire5: B -> A, 256x27x27 -> 256x27x27
  C = run_fire(&B, C, W, &A, f5s, 32, f5e1, 128, f5e3, 128);
  if (C != 256)
    return false;
  print_buffer_memory("after_fire5");
  print_free_memory("after_fire5");

  // Pool3: A -> B, 256x27x27 -> 256x13x13
  W = noodle_pool2d(&A, C, W, &B, 3, 2);
  if (W != 13)
  {
    Serial.print(F("ERR pool3 W="));
    Serial.println(W);
    return false;
  }
  print_buffer_memory("after_pool3");

  // Fire6: B -> A, 256x13x13 -> 384x13x13
  C = run_fire(&B, C, W, &A, f6s, 48, f6e1, 192, f6e3, 192);
  if (C != 384)
    return false;
  print_buffer_memory("after_fire6");

  // Fire7: A -> B, 384x13x13 -> 384x13x13
  C = run_fire(&A, C, W, &B, f7s, 48, f7e1, 192, f7e3, 192);
  if (C != 384)
    return false;
  print_buffer_memory("after_fire7");

  // Fire8: B -> A, 384x13x13 -> 512x13x13
  C = run_fire(&B, C, W, &A, f8s, 64, f8e1, 256, f8e3, 256);
  if (C != 512)
    return false;
  print_buffer_memory("after_fire8");

  // Fire9: A -> B, 512x13x13 -> 512x13x13
  C = run_fire(&A, C, W, &B, f9s, 64, f9e1, 256, f9e3, 256);
  if (C != 512)
    return false;
  print_buffer_memory("after_fire9");
  print_free_memory("after_fire9");

  // Final classifier conv: B -> A, 512x13x13 -> 1000x13x13
  W = noodle_conv_float(&B, C, NUM_CLASSES, &A, W, final_conv, p, NULL);
  if (W != 13)
  {
    Serial.print(F("ERR final_conv W="));
    Serial.println(W);
    return false;
  }
  C = NUM_CLASSES;
  print_buffer_memory("after_final_conv");
  print_free_memory("after_final_conv");

  // GAP in-place on A: 1000x13x13 -> 1000
  uint16_t V = noodle_gap(&A, C, W);
  if (V != C)
  {
    Serial.print(F("ERR gap V="));
    Serial.println(V);
    return false;
  }

  V = noodle_soft_max(&A, NUM_CLASSES);
  if (V != NUM_CLASSES)
  {
    Serial.print(F("ERR softmax V="));
    Serial.println(V);
    return false;
  }

  uint16_t pred = 0;
  float max_val = 0.0f;
  noodle_find_max(&A, NUM_CLASSES, max_val, pred);

  const float et = (float)(micros() - t_all) * 1e-6f;

  char pred_line[96];
  snprintf(pred_line, sizeof(pred_line),
           "PRED %u %.6f %.6f class_%u",
           pred,
           et,
           max_val,
           pred);

  Serial.println(pred_line);
  Serial.flush();

  return true;
}

// ============================================================
// Arduino setup / loop
// ============================================================

void setup()
{
  NoodleSerial::begin(921600);
  NoodleSerial::clear_input();

  noodle_buffer_init(&X);
  noodle_buffer_init(&A);
  noodle_buffer_init(&B);
  noodle_buffer_init(&S);
  noodle_buffer_init(&E1);
  noodle_buffer_init(&E3);

  Serial.println();
  Serial.println(F("BOOT Noodle Full SqueezeNet-1.1 224x224"));
  Serial.println(F("Input: RGB 224x224x3 HWC uint8"));
  Serial.println(F("Runtime: NoodleBuffer + Fire concat + GAP + Softmax"));
  Serial.println(F("Weights: model_weights.h must provide w01..w26 / b01..b26"));

  print_free_memory("boot");

  RX_BYTES = alloc_u8_buffer(IMG_BYTES);
  if (!RX_BYTES)
  {
    Serial.println(F("ERR_ALLOC_RX_BYTES"));
    print_free_memory("alloc_rx_fail");
    while (true)
      delay(1000);
  }

  print_free_memory("after_rx_alloc");
  NoodleSerial::print_ready();
}

void loop()
{
  if (!NoodleSerial::wait_for_img_header())
  {
    NoodleSerial::print_ready();
    return;
  }

  if (!NoodleSerial::recv_image_chunked(RX_BYTES, IMG_BYTES))
  {
    NoodleSerial::print_ready();
    return;
  }

  print_free_memory("after_recv");

  if (!hwc_rgb_u8_to_chw_float(RX_BYTES, &X))
  {
    Serial.println(F("ERR input alloc"));
    print_free_memory("input_alloc_fail");
    NoodleSerial::print_ready();
    return;
  }

  print_buffer_memory("after_input");
  print_free_memory("after_input");

  predict();

  print_buffer_memory("after_predict");
  print_free_memory("after_predict");

  NoodleSerial::print_ready();
}