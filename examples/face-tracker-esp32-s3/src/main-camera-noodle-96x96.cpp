#include <Arduino.h>
#include <Wire.h>
#include <U8g2lib.h>
#include "esp_camera.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "noodle.h"
#include "tinyfacecenter_flatten_weights.h"

// ============================================================
// I2C / OLED
// ============================================================

#define I2C_SDA 11
#define I2C_SCL 10

// 90-degree OLED rotation: logical display becomes 64 x 128.
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R1, U8X8_PIN_NONE);

// ============================================================
// ESP32-S3 Audio Board camera pins
// ============================================================

#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   43
#define CAM_PIN_SIOD   11
#define CAM_PIN_SIOC   10

#define CAM_PIN_D0     2
#define CAM_PIN_D1     17
#define CAM_PIN_D2     18
#define CAM_PIN_D3     39
#define CAM_PIN_D4     45
#define CAM_PIN_D5     46
#define CAM_PIN_D6     47
#define CAM_PIN_D7     48

#define CAM_PIN_VSYNC  21
#define CAM_PIN_HREF   1
#define CAM_PIN_PCLK   44

// ============================================================
// TinyFaceCenter dimensions
// ============================================================

static const uint16_t IMG_W = 96;
static const uint16_t IMG_H = 96;
static const uint16_t IMG_C = 1;
static const size_t   IMG_N = (size_t)IMG_W * IMG_H * IMG_C;

static NoodleBuffer X;
static NoodleBuffer A;
static NoodleBuffer B;

// Latest raw prediction.
// Model output is [p, x01, y01, s].
// x01/y01 are absolute normalized face-center coordinates in the 96x96 frame.
static float g_p   = 0.0f;
static float g_x01 = 0.5f;
static float g_y01 = 0.5f;
static float g_s   = 0.0f;
static float g_last_ms = 0.0f;

// Display coordinates. For debugging we use raw x01/y01 directly.
// Later, a low-pass filter may be reintroduced here if the motor jitters.
static float g_x01_f = 0.5f;
static float g_y01_f = 0.5f;
static bool  g_filter_ready = false;

// ============================================================
// Dual-core AI state
// ============================================================

static portMUX_TYPE ai_mux = portMUX_INITIALIZER_UNLOCKED;

static volatile bool ai_busy = false;
static volatile bool ai_request = false;
static volatile bool ai_has_result = false;

// ============================================================
// Small helpers
// ============================================================

static inline int clampi(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// ============================================================
// Camera + OLED helpers
// ============================================================

static bool init_camera() {
  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0 = CAM_PIN_D0;
  config.pin_d1 = CAM_PIN_D1;
  config.pin_d2 = CAM_PIN_D2;
  config.pin_d3 = CAM_PIN_D3;
  config.pin_d4 = CAM_PIN_D4;
  config.pin_d5 = CAM_PIN_D5;
  config.pin_d6 = CAM_PIN_D6;
  config.pin_d7 = CAM_PIN_D7;

  config.pin_xclk  = CAM_PIN_XCLK;
  config.pin_pclk  = CAM_PIN_PCLK;
  config.pin_vsync = CAM_PIN_VSYNC;
  config.pin_href  = CAM_PIN_HREF;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_pwdn  = CAM_PIN_PWDN;
  config.pin_reset = CAM_PIN_RESET;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;

  // Direct 96 x 96 camera frame.
  // This removes the 160x120 -> center-crop step.
  config.frame_size   = FRAMESIZE_96X96;

  config.jpeg_quality = 12;
  config.fb_count     = 1;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  return err == ESP_OK;
}

static void oled_message(const char *a, const char *b = nullptr) {
  u8g2.clearBuffer();
  u8g2.setDrawColor(1);
  u8g2.setFont(u8g2_font_6x10_tf);
  u8g2.drawStr(0, 14, a);
  if (b) u8g2.drawStr(0, 30, b);
  u8g2.sendBuffer();
}

static void draw_camera_96x96(camera_fb_t *fb) {
  if (!fb || !fb->buf) return;

  const int fw = fb->width;
  const int fh = fb->height;

  u8g2.clearBuffer();
  u8g2.setDrawColor(1);

  if (fw != 96 || fh != 96 || fb->format != PIXFORMAT_GRAYSCALE) {
    u8g2.setFont(u8g2_font_5x8_tf);
    u8g2.drawStr(0, 8,  "BAD FRAME");
    char line[24];
    snprintf(line, sizeof(line), "%dx%d", fw, fh);
    u8g2.drawStr(0, 18, line);
    return;
  }

  // Draw 96x96 camera frame as 64x96 portrait OLED preview.
  // Width is squeezed 96 -> 64; height stays 96.
  for (int oy = 0; oy < 96; oy++) {
    const int sy = oy;

    for (int ox = 0; ox < 64; ox++) {
      const int sx = (ox * 96) / 64;
      const uint8_t g = fb->buf[sy * 96 + sx];

      if (g > 110) {
        u8g2.drawPixel(ox, 32 + oy);
      }
    }
  }

  // Top status area. This will be overwritten by prediction overlay.
  u8g2.setFont(u8g2_font_5x8_tf);
  u8g2.drawStr(0, 8,  "LIVE AI");
  u8g2.drawStr(0, 18, "96 FRAME");
  u8g2.drawStr(0, 28, "NO CROP");

  // Center reference cross.
  const int cx = 32;
  const int cy = 32 + 48;
  u8g2.drawLine(cx - 5, cy, cx + 5, cy);
  u8g2.drawLine(cx, cy - 5, cx, cy + 5);

  // Do not send here. Prediction overlay sends once per frame.
}

static bool camera_gray_96x96_to_input(camera_fb_t *fb, NoodleBuffer *dst) {
  if (!fb || !fb->buf || !dst) return false;
  if (fb->format != PIXFORMAT_GRAYSCALE) return false;
  if (fb->width != 96 || fb->height != 96) return false;

  float *x = noodle_buffer_require(dst, IMG_N);
  if (!x) return false;

  for (int i = 0; i < 96 * 96; i++) {
    x[i] = (float)fb->buf[i] * (1.0f / 255.0f);
  }

  return true;
}

// Copy the current frame into X only if the AI task is idle.
// The AI task owns X/A/B after ai_request becomes true until ai_busy becomes false.
static bool submit_frame_for_ai(camera_fb_t *fb) {
  bool can_submit = false;

  portENTER_CRITICAL(&ai_mux);
  if (!ai_busy && !ai_request) {
    ai_busy = true;
    can_submit = true;
  }
  portEXIT_CRITICAL(&ai_mux);

  if (!can_submit) return false;

  bool ok = camera_gray_96x96_to_input(fb, &X);

  portENTER_CRITICAL(&ai_mux);
  if (ok) {
    ai_request = true;
  } else {
    ai_busy = false;
  }
  portEXIT_CRITICAL(&ai_mux);

  return ok;
}

// ============================================================
// Noodle TinyFaceCenter Flatten
// ============================================================

static void make_conv(ConvMem &c,
                      uint16_t K,
                      uint16_t P,
                      uint16_t S_stride,
                      const NoodleWeight *w,
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

static void make_fcn(FCNMem &f,
                     const NoodleWeight *w,
                     const float *b,
                     Activation act) {
  f.weight = w;
  f.bias = b;
  f.act = act;
}

static Pool pool2x2() {
  Pool p;
  p.M = 2;
  p.T = 2;
  return p;
}

static bool run_tinyfacecenter(float &p_out,
                               float &x01_out,
                               float &y01_out,
                               float &s_out,
                               float &ms_out) {
  const Pool p2 = pool2x2();

  ConvMem c1, c2, c3, c4;
  make_conv(c1, 3, 65535, 1, w01, b01, ACT_RELU);
  make_conv(c2, 3, 65535, 1, w02, b02, ACT_RELU);
  make_conv(c3, 3, 65535, 1, w03, b03, ACT_RELU);
  make_conv(c4, 3, 65535, 1, w04, b04, ACT_RELU);

  FCNMem d1, dout;
  make_fcn(d1,   w05, b05, ACT_RELU);
  make_fcn(dout, w06, b06, ACT_NONE);

  uint16_t W = IMG_W;
  uint16_t C = IMG_C;

  const uint32_t t0 = micros();

  // conv1 + pool1: 1 x 96 x 96 -> 8 x 48 x 48
  W = noodle_conv_float(&X, C, 8, &A, W, c1, p2, NULL);
  if (W != 48) return false;
  C = 8;

  // conv2 + pool2: 8 x 48 x 48 -> 16 x 24 x 24
  W = noodle_conv_float(&A, C, 16, &B, W, c2, p2, NULL);
  if (W != 24) return false;
  C = 16;

  // conv3 + pool3: 16 x 24 x 24 -> 32 x 12 x 12
  W = noodle_conv_float(&B, C, 32, &A, W, c3, p2, NULL);
  if (W != 12) return false;
  C = 32;

  // conv4 + pool4: 32 x 12 x 12 -> 32 x 6 x 6
  W = noodle_conv_float(&A, C, 32, &B, W, c4, p2, NULL);
  if (W != 6) return false;
  C = 32;

  // Flatten: Noodle CHW 32 x 6 x 6 -> Keras-compatible HWC flat vector.
  // This preserves spatial location, unlike GAP.
  uint16_t V = noodle_flat(&B, &A, W, C);
  if (V != 1152) return false;

  // Dense1: 1152 -> 64
  V = noodle_fcn(&A, 1152, 64, &B, d1, NULL);
  if (V != 64) return false;

  // Output dense: 64 -> 4, then sigmoid.
  V = noodle_fcn(&B, 64, 4, &A, dout, NULL);
  if (V != 4) return false;

  V = noodle_sigmoid(&A, 4);
  if (V != 4) return false;

  ms_out = (float)(micros() - t0) * 1e-3f;

  float *o = A.data;

  // New model output:
  //   o[0] = p
  //   o[1] = x01, absolute face-center x coordinate, 0..1
  //   o[2] = y01, absolute face-center y coordinate, 0..1
  //   o[3] = s, relative face size
  p_out   = o[0];
  x01_out = o[1];
  y01_out = o[2];
  s_out   = o[3];

  return true;
}

static void noodle_ai_task(void *param) {
  (void)param;

  while (true) {
    bool do_run = false;

    portENTER_CRITICAL(&ai_mux);
    if (ai_request) {
      ai_request = false;
      do_run = true;
    }
    portEXIT_CRITICAL(&ai_mux);

    if (do_run) {
      float p, x01, y01, s, ms;
      bool ok = run_tinyfacecenter(p, x01, y01, s, ms);

      portENTER_CRITICAL(&ai_mux);

      if (ok) {
        g_p = p;
        g_x01 = x01;
        g_y01 = y01;
        g_s = s;
        g_last_ms = ms;

        // For debugging, show the raw coordinate immediately.
        // Later, for motor control, we may smooth x01/y01 or dx/dy here.
        g_x01_f = g_x01;
        g_y01_f = g_y01;
        g_filter_ready = true;

        ai_has_result = true;
      }

      ai_busy = false;
      portEXIT_CRITICAL(&ai_mux);
    }

    vTaskDelay(pdMS_TO_TICKS(1));
  }
}


// ============================================================
// Result overlay
// ============================================================

static void draw_white_crosshair(int cx, int cy) {
  const int r = 16;   // longer
  const int t = 2;    // half-thickness; total thickness = 5 pixels

  u8g2.setDrawColor(1);

  // Thick horizontal line.
  for (int yy = cy - t; yy <= cy + t; yy++) {
    if (yy < 32 || yy >= 128) continue;
    u8g2.drawLine(clampi(cx - r, 0, 63), yy, clampi(cx + r, 0, 63), yy);
  }

  // Thick vertical line.
  for (int xx = cx - t; xx <= cx + t; xx++) {
    if (xx < 0 || xx >= 64) continue;
    u8g2.drawLine(xx, clampi(cy - r, 32, 127), xx, clampi(cy + r, 32, 127));
  }
}

static void draw_face_center_marker(int cx, int cy) {
  // Clear a small black square first so marker is visible
  u8g2.setDrawColor(0);
  u8g2.drawBox(max(0, cx - 6), max(32, cy - 6),
               min(13, 64 - max(0, cx - 6)),
               min(13, 128 - max(32, cy - 6)));

  // Draw white target circle + center dot
  u8g2.setDrawColor(1);
  u8g2.drawCircle(cx, cy, 5);
  u8g2.drawDisc(cx, cy, 2);
}

static void draw_prediction_overlay(float p, float x01, float y01, float s) {
  // x01/y01 are absolute normalized face-center coordinates.
  // They are NOT dx/dy. For motor later:
  //   dx = x01 - 0.5
  //   dy = y01 - 0.5
  if (x01 < 0.0f) x01 = 0.0f;
  if (x01 > 1.0f) x01 = 1.0f;
  if (y01 < 0.0f) y01 = 0.0f;
  if (y01 > 1.0f) y01 = 1.0f;

  const float dx = x01 - 0.5f;
  const float dy = y01 - 0.5f;

  // OLED preview is 64 x 96, starting at y=32.
  // Camera frame is 96 x 96, squeezed horizontally to 64 on the OLED.
  int cx = (int)roundf(x01 * 63.0f);
  int cy = 32 + (int)roundf(y01 * 95.0f);

  // Predicted face center first.
  draw_face_center_marker(cx, cy);

  // Fixed image-center reference last.
  const int center_x = 32;
  const int center_y = 32 + 48;

  u8g2.setDrawColor(1);
  u8g2.drawLine(center_x - 4, center_y, center_x + 4, center_y);
  u8g2.drawLine(center_x, center_y - 4, center_x, center_y + 4);

  // Top black text panel with white text.
  u8g2.setDrawColor(0);
  u8g2.drawBox(0, 0, 64, 31);
  u8g2.setDrawColor(1);

  char line[24];
  u8g2.setFont(u8g2_font_5x8_tf);

  snprintf(line, sizeof(line), "p%.2f %.0fms", p, g_last_ms);
  u8g2.drawStr(0, 8, line);

  snprintf(line, sizeof(line), "x%.2f y%.2f", x01, y01);
  u8g2.drawStr(0, 18, line);

  snprintf(line, sizeof(line), "dx%+.2f dy%+.2f", dx, dy);
  u8g2.drawStr(0, 28, line);

  u8g2.sendBuffer();
}

// ============================================================
// Arduino setup / loop
// ============================================================

void setup() {
  Serial.begin(921600);
  delay(1000);

  // Camera first. It uses the same SCCB/I2C pins during initialization.
  if (!init_camera()) {
    while (true) delay(1000);
  }

  // Reinitialize I2C for OLED after camera init.
  Wire.end();
  delay(50);
  Wire.begin(I2C_SDA, I2C_SCL);

  // Use 400 kHz for faster OLED refresh. If unstable, change to 100000.
  Wire.setClock(400000);

  u8g2.begin();
  u8g2.setPowerSave(0);
  oled_message("CAMERA OK", "FACECENTER READY");

  noodle_buffer_init(&X);
  noodle_buffer_init(&A);
  noodle_buffer_init(&B);

  // Pre-allocate buffers so they do not grow during the live loop.
  noodle_buffer_require(&X, IMG_N);                    // 1 x 96 x 96
  noodle_buffer_require(&A, (size_t)8 * 48 * 48);      // largest A: conv1 output
  noodle_buffer_require(&B, (size_t)16 * 24 * 24);     // largest B: conv2 output

  // Arduino loop typically runs on core 1.
  // Put Noodle inference on core 0.
  xTaskCreatePinnedToCore(
    noodle_ai_task,
    "noodle_ai",
    12288, // 12KB general-purpose FreeRTOS task stack
    NULL,
    1,
    NULL,
    0
  );

  delay(500);
}

void loop() {
  static uint32_t last_print_ms = 0;

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    oled_message("Frame FAIL");
    delay(100);
    return;
  }

  // Submit frame to AI only if the AI task is idle.
  // This copies the 96x96 grayscale frame into X.
  submit_frame_for_ai(fb);

  // Draw the live video frame every loop.
  draw_camera_96x96(fb);

  esp_camera_fb_return(fb);
  fb = nullptr;

  // Copy the latest prediction safely.
  float p, x01, y01, s, raw_x01, raw_y01, ms;
  bool has_result;
  bool busy;

  portENTER_CRITICAL(&ai_mux);
  p = g_p;
  x01 = g_x01_f;
  y01 = g_y01_f;
  raw_x01 = g_x01;
  raw_y01 = g_y01;
  s = g_s;
  ms = g_last_ms;
  has_result = ai_has_result;
  busy = ai_busy || ai_request;
  portEXIT_CRITICAL(&ai_mux);

  if (has_result) {
    draw_prediction_overlay(p, x01, y01, s);
  } else {
    u8g2.sendBuffer();
  }

  uint32_t now = millis();
  if (now - last_print_ms >= 500) {
    last_print_ms = now;
    Serial.printf(
      "FACE p=%.4f x=%.4f y=%.4f dx=%+.4f dy=%+.4f s=%.4f t=%.2fms busy=%d\n",
      p, x01, y01, x01 - 0.5f, y01 - 0.5f, s, ms, busy ? 1 : 0
    );
  }
}
