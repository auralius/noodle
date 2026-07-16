#include <Arduino.h>
#include <stdlib.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include "esp_camera.h"
#include "camera_pins.h"
#include "imagenet_classes.h"

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
#include "model_weights.h"

// ============================================================
// ESP32-S3 Camera + TFT Viewfinder + Button-triggered SqueezeNet
// ============================================================
//
// Hardware:
//   Camera: ESP32-S3 camera pins from camera_pins.h
//   TFT:    ST7735 128x160 SPI TFT
//   Button: GPIO1 -> button -> GND, using INPUT_PULLUP
//
// Workflow:
//   1. Live QVGA RGB565 camera preview on TFT.
//   2. Red rectangle shows the 224x224 SqueezeNet crop.
//   3. Press button on GPIO1 to freeze current frame.
//   4. Current QVGA frame is center-cropped to 224x224.
//   5. Crop is converted directly to CHW float ImageNet-normalized tensor.
//   6. Camera frame is returned before SqueezeNet inference.
//   7. SqueezeNet runs and prediction is shown on TFT + Serial.
//
// Required model_weights.h:
//   must define w01..w26 and b01..b26 for SqueezeNet-1.1 layer order.

// ============================================================
// TFT pin mapping
// ============================================================

#define TFT_SCLK  39
#define TFT_MOSI  40
#define TFT_CS    38
#define TFT_DC    41
#define TFT_RST   42
#define TFT_MISO  -1

Adafruit_ST7735 tft = Adafruit_ST7735(&SPI, TFT_CS, TFT_DC, TFT_RST);

// ============================================================
// Button
// ============================================================

#define BUTTON_PIN 1

// ============================================================
// Camera / preview settings
// ============================================================

#define SERIAL_BAUD 921600

// Camera: QVGA RGB565 = 320x240
static const framesize_t CAM_FRAME_SIZE = FRAMESIZE_QVGA;

// Current working TFT orientation is landscape 160x128.
// Preview uses 160x120, bottom 8 pixels for status.
static const int PREVIEW_W = 160;
static const int PREVIEW_H = 120;
static const int PREVIEW_X = 0;
static const int PREVIEW_Y = 0;

static uint16_t linebuf[PREVIEW_W];

// Required for your current camera/TFT combination.
#define SWAP_RGB565_BYTES 1

// ============================================================
// SqueezeNet input / output
// ============================================================

#define PRINT_MEM_DEBUG 1

static const uint16_t IMG_W = 224;
static const uint16_t IMG_H = 224;
static const uint16_t IMG_C = 3;
static const uint32_t IMG_PIXELS = (uint32_t)IMG_W * IMG_H;
static const uint32_t IMG_BYTES = IMG_PIXELS * IMG_C;
static const uint16_t NUM_CLASSES = 1000;

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

static uint16_t g_last_pred = 0;
static float g_last_conf = 0.0f;
static float g_last_sec = 0.0f;

// ============================================================
// Small display helpers
// ============================================================

static inline uint16_t maybe_swap565(uint16_t c) {
#if SWAP_RGB565_BYTES
  return (c >> 8) | (c << 8);
#else
  return c;
#endif
}

static void draw_status_line(const char *msg, uint16_t color = ST77XX_GREEN) {
  tft.fillRect(0, 120, 160, 8, ST77XX_BLACK);
  tft.setCursor(2, 121);
  tft.setTextSize(1);
  tft.setTextColor(color, ST77XX_BLACK);
  tft.print(msg);
}

static void draw_status(float fps, uint32_t frame_count) {
  tft.fillRect(0, 120, 160, 8, ST77XX_BLACK);
  tft.setCursor(2, 121);
  tft.setTextColor(ST77XX_GREEN, ST77XX_BLACK);
  tft.printf("FPS %.1f  F%lu", fps, (unsigned long)frame_count);
}

static void draw_running_ui() {
  tft.fillRect(0, 120, 160, 8, ST77XX_BLACK);
  tft.setCursor(2, 121);
  tft.setTextColor(ST77XX_YELLOW, ST77XX_BLACK);
  tft.print("RUNNING...");
}

static void draw_prediction_ui(uint16_t pred, float conf, float sec) {
  tft.fillScreen(ST77XX_BLACK);

  tft.setTextWrap(false);
  tft.setTextSize(1);

  tft.setTextColor(ST77XX_CYAN, ST77XX_BLACK);
  tft.setCursor(2, 4);
  tft.print("SqueezeNet-1.1");

  tft.setTextColor(ST77XX_WHITE, ST77XX_BLACK);
  tft.setCursor(2, 24);
  tft.printf("Class: %u", pred);

  tft.setCursor(2, 40);
  tft.print(imagenet_label(pred));

  tft.setCursor(2, 56);
  tft.printf("Conf : %.4f", conf);

  tft.setCursor(2, 72);
  tft.printf("Time : %.1f s", sec);

  tft.setTextColor(ST77XX_GREEN, ST77XX_BLACK);
  tft.setCursor(2, 96);
  tft.print("Press button");

  tft.setCursor(2, 110);
  tft.print("for preview");
}

// ============================================================
// Memory helpers
// ============================================================

static int free_ram_approx() {
#if defined(ARDUINO_ARCH_ESP32)
  return (int)ESP.getFreeHeap();
#else
  return -1;
#endif
}

static float fragmentation_percent(size_t free_bytes, size_t largest_block) {
  if (free_bytes == 0) return 0.0f;

  float frag = 100.0f * (1.0f - ((float)largest_block / (float)free_bytes));
  if (frag < 0.0f) frag = 0.0f;
  if (frag > 100.0f) frag = 100.0f;

  return frag;
}

static void print_free_memory(const char *tag) {
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

static void print_buffer_memory(const char *tag) {
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
// Button debounce
// ============================================================

static bool button_pressed_once() {
  static bool last_raw = HIGH;
  static bool stable_state = HIGH;
  static bool was_pressed = false;
  static uint32_t last_change_ms = 0;

  bool raw = digitalRead(BUTTON_PIN);
  uint32_t now = millis();

  if (raw != last_raw) {
    last_raw = raw;
    last_change_ms = now;
  }

  if ((now - last_change_ms) > 30 && stable_state != raw) {
    stable_state = raw;
  }

  bool is_pressed = (stable_state == LOW);
  bool event = is_pressed && !was_pressed;
  was_pressed = is_pressed;

  return event;
}

// ============================================================
// Camera/TFT init
// ============================================================

static bool init_camera() {
  camera_config_t config;
  cam_fill_pins(config);

  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size   = CAM_FRAME_SIZE;
  config.fb_count     = 1;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("CAM_INIT_FAIL 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 0);
    s->set_contrast(s, 1);
    s->set_saturation(s, 0);

    // Your current working orientation:
    s->set_hmirror(s, 0);
    s->set_vflip(s, 1);
  }

  Serial.println("CAM_INIT_OK");
  return true;
}

static void init_tft() {
  SPI.begin(TFT_SCLK, TFT_MISO, TFT_MOSI, TFT_CS);

  // If display offsets are wrong, try INITR_GREENTAB or INITR_REDTAB.
  tft.initR(INITR_BLACKTAB);

  // Your current working orientation.
  tft.setRotation(1);

  tft.fillScreen(ST77XX_BLACK);
  tft.setTextWrap(false);
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_GREEN, ST77XX_BLACK);
  tft.setCursor(2, 122);
  tft.print("TFT OK");
}

// ============================================================
// Draw half-resolution camera preview
// ============================================================

static void draw_camera_preview_half(const camera_fb_t *fb) {
  if (!fb || !fb->buf) return;

  if (fb->format != PIXFORMAT_RGB565) {
    Serial.println("ERR_NOT_RGB565");
    return;
  }

  const int src_w = fb->width;   // expected 320
  const int src_h = fb->height;  // expected 240

  const uint16_t *src = (const uint16_t *)fb->buf;

  // Downsample 320x240 -> 160x120 by taking every 2nd pixel.
  for (int y = 0; y < PREVIEW_H; y++) {
    int sy = y * 2;
    if (sy >= src_h) sy = src_h - 1;

    const uint16_t *src_row = src + sy * src_w;

    for (int x = 0; x < PREVIEW_W; x++) {
      int sx = x * 2;
      if (sx >= src_w) sx = src_w - 1;

      linebuf[x] = maybe_swap565(src_row[sx]);
    }

    tft.drawRGBBitmap(PREVIEW_X, PREVIEW_Y + y, linebuf, PREVIEW_W, 1);
  }

  // Draw SqueezeNet 224x224 crop guide.
  // In QVGA: crop x=48..271, y=8..231.
  // In half preview: x=24..135, y=4..115.
  const int guide_x = 24;
  const int guide_y = 4;
  const int guide_w = 112;
  const int guide_h = 112;

  tft.drawRect(guide_x, guide_y, guide_w, guide_h, ST77XX_RED);
}

// ============================================================
// Convert current camera frame to SqueezeNet input
// ============================================================

static bool camera_rgb565_qvga_crop224_to_chw_float(const camera_fb_t *fb, NoodleBuffer *dst) {
  if (!fb || !fb->buf || !dst) return false;
  if (fb->format != PIXFORMAT_RGB565) return false;

  const int src_w = fb->width;   // expected 320
  const int src_h = fb->height;  // expected 240

  if (src_w < IMG_W || src_h < IMG_H) return false;

  float *x = noodle_buffer_require(dst, (size_t)IMG_PIXELS * IMG_C);
  if (!x) return false;

  const int crop_x = (src_w - IMG_W) / 2; // 48 for QVGA
  const int crop_y = (src_h - IMG_H) / 2; // 8  for QVGA

  const uint16_t *src = (const uint16_t *)fb->buf;

  for (uint16_t y = 0; y < IMG_H; ++y) {
    const int sy = crop_y + y;
    const uint16_t *row = src + sy * src_w;

    for (uint16_t xpix = 0; xpix < IMG_W; ++xpix) {
      const int sx = crop_x + xpix;

      // Camera frame is already in memory order used earlier.
      // For numeric RGB extraction, apply the same byte-swap as preview first.
      uint16_t p = maybe_swap565(row[sx]);

      uint8_t r5 = (p >> 11) & 0x1F;
      uint8_t g6 = (p >> 5)  & 0x3F;
      uint8_t b5 =  p        & 0x1F;

      float r = (float)r5 * (255.0f / 31.0f);
      float g = (float)g6 * (255.0f / 63.0f);
      float b = (float)b5 * (255.0f / 31.0f);

      const uint32_t chw0 = (uint32_t)y * IMG_W + xpix;

      // PyTorch SqueezeNet/ImageNet preprocessing:
      x[0 * IMG_PIXELS + chw0] = (r * (1.0f / 255.0f) - 0.485f) / 0.229f;
      x[1 * IMG_PIXELS + chw0] = (g * (1.0f / 255.0f) - 0.456f) / 0.224f;
      x[2 * IMG_PIXELS + chw0] = (b * (1.0f / 255.0f) - 0.406f) / 0.225f;
    }
  }

  return true;
}

// ============================================================
// Noodle SqueezeNet helpers
// ============================================================

static void make_conv(ConvMem &c,
                      uint16_t K,
                      uint16_t P,
                      uint16_t S_stride,
                      const float *w,
                      const float *b,
                      Activation act) {
  c.K = K;
  c.P = P;
  c.S = S_stride;
  c.OP = 0;
  c.weight = w;
  c.bias = b;
  c.act = act;
}

static Pool no_pool() {
  Pool p;
  p.M = 1;
  p.T = 1;
  return p;
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
                         uint16_t C_e3) {
  const Pool p = no_pool();

  uint16_t V = noodle_conv_float(input, C_in, C_sq, &S, W, squeeze, p, NULL);
  if (V != W) {
    Serial.print(F("ERR fire squeeze W="));
    Serial.println(V);
    return 0;
  }

  V = noodle_conv_float(&S, C_sq, C_e1, &E1, W, expand1, p, NULL);
  if (V != W) {
    Serial.print(F("ERR fire expand1 W="));
    Serial.println(V);
    return 0;
  }

  V = noodle_conv_float(&S, C_sq, C_e3, &E3, W, expand3, p, NULL);
  if (V != W) {
    Serial.print(F("ERR fire expand3 W="));
    Serial.println(V);
    return 0;
  }

  const uint16_t C_out = noodle_concat(&E1, C_e1, &E3, C_e3, output, W);
  if (C_out != (uint16_t)(C_e1 + C_e3)) {
    Serial.println(F("ERR fire concat"));
    return 0;
  }

  return C_out;
}

// ============================================================
// Prediction
// ============================================================

static bool predict() {
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
  if (W != 111) {
    Serial.print(F("ERR conv1 W="));
    Serial.println(W);
    return false;
  }
  C = 64;
  print_buffer_memory("after_conv1");
  print_free_memory("after_conv1");

  // Pool1: A -> B, 64x111x111 -> 64x55x55
  W = noodle_pool2d(&A, C, W, &B, 3, 2);
  if (W != 55) {
    Serial.print(F("ERR pool1 W="));
    Serial.println(W);
    return false;
  }
  print_buffer_memory("after_pool1");

  // Fire2: B -> A, 64x55x55 -> 128x55x55
  C = run_fire(&B, C, W, &A, f2s, 16, f2e1, 64, f2e3, 64);
  if (C != 128) return false;
  print_buffer_memory("after_fire2");
  print_free_memory("after_fire2");

  // Fire3: A -> B, 128x55x55 -> 128x55x55
  C = run_fire(&A, C, W, &B, f3s, 16, f3e1, 64, f3e3, 64);
  if (C != 128) return false;
  print_buffer_memory("after_fire3");

  // Pool2: B -> A, 128x55x55 -> 128x27x27
  W = noodle_pool2d(&B, C, W, &A, 3, 2);
  if (W != 27) {
    Serial.print(F("ERR pool2 W="));
    Serial.println(W);
    return false;
  }
  print_buffer_memory("after_pool2");
  print_free_memory("after_pool2");

  // Fire4: A -> B, 128x27x27 -> 256x27x27
  C = run_fire(&A, C, W, &B, f4s, 32, f4e1, 128, f4e3, 128);
  if (C != 256) return false;
  print_buffer_memory("after_fire4");

  // Fire5: B -> A, 256x27x27 -> 256x27x27
  C = run_fire(&B, C, W, &A, f5s, 32, f5e1, 128, f5e3, 128);
  if (C != 256) return false;
  print_buffer_memory("after_fire5");
  print_free_memory("after_fire5");

  // Pool3: A -> B, 256x27x27 -> 256x13x13
  W = noodle_pool2d(&A, C, W, &B, 3, 2);
  if (W != 13) {
    Serial.print(F("ERR pool3 W="));
    Serial.println(W);
    return false;
  }
  print_buffer_memory("after_pool3");

  // Fire6: B -> A, 256x13x13 -> 384x13x13
  C = run_fire(&B, C, W, &A, f6s, 48, f6e1, 192, f6e3, 192);
  if (C != 384) return false;
  print_buffer_memory("after_fire6");

  // Fire7: A -> B, 384x13x13 -> 384x13x13
  C = run_fire(&A, C, W, &B, f7s, 48, f7e1, 192, f7e3, 192);
  if (C != 384) return false;
  print_buffer_memory("after_fire7");

  // Fire8: B -> A, 384x13x13 -> 512x13x13
  C = run_fire(&B, C, W, &A, f8s, 64, f8e1, 256, f8e3, 256);
  if (C != 512) return false;
  print_buffer_memory("after_fire8");

  // Fire9: A -> B, 512x13x13 -> 512x13x13
  C = run_fire(&A, C, W, &B, f9s, 64, f9e1, 256, f9e3, 256);
  if (C != 512) return false;
  print_buffer_memory("after_fire9");
  print_free_memory("after_fire9");

  // Final classifier conv: B -> A, 512x13x13 -> 1000x13x13
  W = noodle_conv_float(&B, C, NUM_CLASSES, &A, W, final_conv, p, NULL);
  if (W != 13) {
    Serial.print(F("ERR final_conv W="));
    Serial.println(W);
    return false;
  }
  C = NUM_CLASSES;
  print_buffer_memory("after_final_conv");
  print_free_memory("after_final_conv");

  // GAP in-place on A: 1000x13x13 -> 1000
  uint16_t V = noodle_gap(&A, C, W);
  if (V != C) {
    Serial.print(F("ERR gap V="));
    Serial.println(V);
    return false;
  }

  V = noodle_soft_max(&A, NUM_CLASSES);
  if (V != NUM_CLASSES) {
    Serial.print(F("ERR softmax V="));
    Serial.println(V);
    return false;
  }

  uint16_t pred = 0;
  float max_val = 0.0f;
  noodle_find_max(&A, NUM_CLASSES, max_val, pred);

  const float et = (float)(micros() - t_all) * 1e-6f;

  g_last_pred = pred;
  g_last_conf = max_val;
  g_last_sec = et;

  char pred_line[96];
  snprintf(pred_line, sizeof(pred_line),
           "PRED %u %.6f %.6f %s",
           pred,
           et,
           max_val,
           imagenet_label(pred));

  Serial.println(pred_line);
  Serial.flush();

  return true;
}

// ============================================================
// Arduino setup / loop
// ============================================================

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);

  Serial.println();
  Serial.println(F("BOOT Camera + TFT + Noodle SqueezeNet-1.1"));
  Serial.println(F("Button: GPIO1 -> button -> GND"));
  Serial.println(F("Input: camera QVGA RGB565 -> center crop 224x224"));
  Serial.println(F("Weights: model_weights.h must provide w01..w26 / b01..b26"));

  pinMode(BUTTON_PIN, INPUT_PULLUP);

  if (psramFound()) {
    Serial.println(F("PSRAM_OK"));
  } else {
    Serial.println(F("WARN_NO_PSRAM"));
  }

  init_tft();

  if (!init_camera()) {
    tft.fillScreen(ST77XX_BLACK);
    tft.setTextColor(ST77XX_RED, ST77XX_BLACK);
    tft.setCursor(0, 20);
    tft.println("CAM FAIL");
    while (true) delay(1000);
  }

  noodle_buffer_init(&X);
  noodle_buffer_init(&A);
  noodle_buffer_init(&B);
  noodle_buffer_init(&S);
  noodle_buffer_init(&E1);
  noodle_buffer_init(&E3);

  print_free_memory("boot");

  tft.fillScreen(ST77XX_BLACK);
  draw_status_line("READY");
  Serial.println(F("READY"));
}

void loop() {
  static uint32_t frame_count = 0;
  static uint32_t last_fps_ms = millis();
  static uint32_t fps_count = 0;
  static float fps = 0.0f;
  static bool showing_prediction = false;

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("CAM_CAPTURE_FAIL");
    draw_status_line("CAM FAIL", ST77XX_RED);
    delay(100);
    return;
  }

  // Live preview while not running inference.
  draw_camera_preview_half(fb);

  bool capture_now = button_pressed_once();

  if (capture_now) {
    Serial.println(F("BUTTON_CAPTURE"));
    draw_running_ui();

    print_free_memory("before_input");

    bool ok = camera_rgb565_qvga_crop224_to_chw_float(fb, &X);

    // Important: return camera frame before SqueezeNet inference.
    esp_camera_fb_return(fb);
    fb = nullptr;

    if (!ok) {
      Serial.println(F("ERR_CAMERA_TO_INPUT"));
      draw_status_line("INPUT ERR", ST77XX_RED);
      delay(1000);
      return;
    }

    print_buffer_memory("after_input");
    print_free_memory("after_input");

    bool pred_ok = predict();

    print_buffer_memory("after_predict");
    print_free_memory("after_predict");

    if (pred_ok) {
      draw_prediction_ui(g_last_pred, g_last_conf, g_last_sec);
      showing_prediction = true;
    } else {
      draw_status_line("PRED ERR", ST77XX_RED);
      delay(1000);
    }

    // Wait until the button is released before returning to preview.
    while (digitalRead(BUTTON_PIN) == LOW) {
      delay(10);
    }

    // Keep prediction visible until next button press.
    // Press again to return to preview.
    while (true) {
      if (button_pressed_once()) {
        tft.fillScreen(ST77XX_BLACK);
        showing_prediction = false;
        break;
      }
      delay(20);
    }

    return;
  }

  if (fb) {
    esp_camera_fb_return(fb);
  }

  frame_count++;
  fps_count++;

  uint32_t now = millis();
  if (now - last_fps_ms >= 1000) {
    fps = fps_count * 1000.0f / (now - last_fps_ms);
    fps_count = 0;
    last_fps_ms = now;

    Serial.printf("FPS %.2f frame=%lu\n", fps, (unsigned long)frame_count);
  }

  if (!showing_prediction) {
    draw_status(fps, frame_count);
  }
}
