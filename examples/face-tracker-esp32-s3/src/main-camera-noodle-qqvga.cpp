#include <Arduino.h>
#include <Wire.h>
#include <U8g2lib.h>
#include "esp_camera.h"

#include "noodle.h"
#include "tinyfacestepper_weights.h"

#define I2C_SDA 11
#define I2C_SCL 10

// 90-degree OLED rotation: logical display is 64 x 128.
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R1, U8X8_PIN_NONE);

// ESP32-S3 Audio Board camera pins
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

// TCA9555 I/O expander for built-in keys
#define TCA_ADDR 0x20
#define TCA_INPUT_PORT1   0x01
#define TCA_CONFIG_PORT1  0x07
#define KEY1_BIT 1  // EXIO9 = P11, active-low
#define KEY2_BIT 2
#define KEY3_BIT 3

// TinyFaceStepper dimensions
static const uint16_t IMG_W = 96;
static const uint16_t IMG_H = 96;
static const uint16_t IMG_C = 1;

static NoodleBuffer X;
static NoodleBuffer A;
static NoodleBuffer B;

static float g_p  = 0.0f;
static float g_dx = 0.0f;
static float g_dy = 0.0f;
static float g_s  = 0.0f;
static float g_last_ms = 0.0f;

// Low-pass filtered output for a steadier live crosshair.
static float g_dx_f = 0.0f;
static float g_dy_f = 0.0f;
static bool  g_filter_ready = false;

// ============================================================
// Dual-core AI task synchronization
// ============================================================
//
// Main loop/core:
//   - captures camera frame
//   - draws OLED preview
//   - copies one 96x96 crop to X only when AI is idle
//
// AI task/core:
//   - consumes X
//   - runs Noodle using X/A/B
//   - publishes the latest p/dx/dy/s result
//
// Important: only the AI task runs Noodle. The main loop never touches
// A/B and only writes X when the AI task is idle.
static portMUX_TYPE ai_mux = portMUX_INITIALIZER_UNLOCKED;

static volatile bool ai_busy       = false;
static volatile bool ai_request    = false;
static volatile bool ai_has_result = false;

// ============================================================
// TCA9555 key helpers
// ============================================================

static uint8_t tca_read_reg(uint8_t reg) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;

  Wire.requestFrom(TCA_ADDR, (uint8_t)1);
  if (Wire.available()) return Wire.read();
  return 0xFF;
}

static void tca_write_reg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

static void init_keys() {
  uint8_t cfg1 = tca_read_reg(TCA_CONFIG_PORT1);
  cfg1 |= (1 << KEY1_BIT) | (1 << KEY2_BIT) | (1 << KEY3_BIT);
  tca_write_reg(TCA_CONFIG_PORT1, cfg1);
}

static bool key1_raw_pressed() {
  uint8_t p1 = tca_read_reg(TCA_INPUT_PORT1);
  return ((p1 & (1 << KEY1_BIT)) == 0);  // active-low
}

static bool key1_pressed_once() {
  static bool last_raw = false;
  static bool stable_state = false;
  static bool was_pressed = false;
  static uint32_t last_change_ms = 0;

  bool raw = key1_raw_pressed();
  uint32_t now = millis();

  if (raw != last_raw) {
    last_raw = raw;
    last_change_ms = now;
  }

  if ((now - last_change_ms) > 30 && stable_state != raw) {
    stable_state = raw;
  }

  bool event = stable_state && !was_pressed;
  was_pressed = stable_state;
  return event;
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
  config.frame_size   = FRAMESIZE_QQVGA;  // 160 x 120
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

static void draw_camera_crop(camera_fb_t *fb) {
  const int fw = fb->width;   // 160
  const int fh = fb->height;  // 120

  const int crop = 96;
  const int x0 = (fw - crop) / 2;
  const int y0 = (fh - crop) / 2;

  u8g2.clearBuffer();
  u8g2.setDrawColor(1);

  // Draw 64 x 96 portrait preview from 96 x 96 crop.
  for (int oy = 0; oy < 96; oy++) {
    int sy = y0 + (oy * crop) / 96;

    for (int ox = 0; ox < 64; ox++) {
      int sx = x0 + (ox * crop) / 64;
      uint8_t g = fb->buf[sy * fw + sx];

      if (g > 110) {
        u8g2.drawPixel(ox, 32 + oy);
      }
    }
  }

  // Top status area. This will be overwritten by the prediction overlay.
  u8g2.setFont(u8g2_font_5x8_tf);
  u8g2.drawStr(0, 8,  "LIVE AI");
  u8g2.drawStr(0, 18, "96 CROP");
  u8g2.drawStr(0, 28, "NOODLE");

  // White center reference cross.
  const int cx = 32;
  const int cy = 32 + 48;
  u8g2.drawLine(cx - 5, cy, cx + 5, cy);
  u8g2.drawLine(cx, cy - 5, cx, cy + 5);

  // Do not send here. The live inference overlay will send once per frame.
}

static bool camera_gray_crop96_to_input(camera_fb_t *fb, NoodleBuffer *dst) {
  if (!fb || !fb->buf || !dst) return false;
  if (fb->format != PIXFORMAT_GRAYSCALE) return false;

  const int fw = fb->width;
  const int fh = fb->height;
  const int crop = 96;
  const int x0 = (fw - crop) / 2;
  const int y0 = (fh - crop) / 2;

  float *x = noodle_buffer_require(dst, (size_t)IMG_W * IMG_H * IMG_C);
  if (!x) return false;

  // Packed CHW. For C=1 this is simply y*96+x.
  for (int y = 0; y < crop; y++) {
    int sy = y0 + y;
    for (int xpix = 0; xpix < crop; xpix++) {
      int sx = x0 + xpix;
      uint8_t g = fb->buf[sy * fw + sx];
      x[y * crop + xpix] = (float)g * (1.0f / 255.0f);
    }
  }

  return true;
}

// ============================================================
// Noodle TinyFaceStepper
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

static bool run_tinyfacestepper(float &p_out,
                                float &dx_out,
                                float &dy_out,
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

  // GAP: 32 x 6 x 6 -> 32
  uint16_t V = noodle_gap(&B, C, W);
  if (V != 32) return false;

  // Dense1: 32 -> 32
  V = noodle_fcn(&B, 32, 32, &A, d1, NULL);
  if (V != 32) return false;

  // Output dense: 32 -> 4, then sigmoid.
  V = noodle_fcn(&A, 32, 4, &B, dout, NULL);
  if (V != 4) return false;

  V = noodle_sigmoid(&B, 4);
  if (V != 4) return false;

  ms_out = (float)(micros() - t0) * 1e-3f;

  float *o = B.data;
  p_out  = o[0];
  dx_out = o[1] - 0.5f;
  dy_out = o[2] - 0.5f;
  s_out  = o[3];

  return true;
}


// ============================================================
// Dual-core AI submit + task
// ============================================================

static bool submit_frame_for_ai(camera_fb_t *fb) {
  bool can_submit = false;

  // Reserve the input buffer only if the AI task is fully idle.
  portENTER_CRITICAL(&ai_mux);
  if (!ai_busy && !ai_request) {
    ai_busy = true;
    can_submit = true;
  }
  portEXIT_CRITICAL(&ai_mux);

  if (!can_submit) {
    return false;  // AI is still working on a previous frame.
  }

  // Main loop writes X only while AI is reserved/idle.
  bool ok = camera_gray_crop96_to_input(fb, &X);

  portENTER_CRITICAL(&ai_mux);
  if (ok) {
    ai_request = true;  // AI task may now consume X.
  } else {
    ai_busy = false;
  }
  portEXIT_CRITICAL(&ai_mux);

  return ok;
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
      float p, dx, dy, s, ms;
      bool ok = run_tinyfacestepper(p, dx, dy, s, ms);

      portENTER_CRITICAL(&ai_mux);

      if (ok) {
        g_p = p;
        g_dx = dx;
        g_dy = dy;
        g_s = s;
        g_last_ms = ms;

        const float alpha = 0.35f;  // larger = faster crosshair response

        if (!g_filter_ready) {
          g_dx_f = g_dx;
          g_dy_f = g_dy;
          g_filter_ready = true;
        } else {
          g_dx_f = (1.0f - alpha) * g_dx_f + alpha * g_dx;
          g_dy_f = (1.0f - alpha) * g_dy_f + alpha * g_dy;
        }

        ai_has_result = true;
      }

      ai_busy = false;
      portEXIT_CRITICAL(&ai_mux);
    }

    // Yield so the camera/OLED loop remains responsive.
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

// ============================================================
// Result overlay
// ============================================================

static void draw_black_crosshair(int cx, int cy) {
  const int r = 9;

  u8g2.setDrawColor(0);

  // Thick horizontal line.
  for (int yy = cy - 1; yy <= cy + 1; yy++) {
    if (yy < 32 || yy >= 128) continue;
    u8g2.drawLine(max(0, cx - r), yy, min(63, cx + r), yy);
  }

  // Thick vertical line.
  for (int xx = cx - 1; xx <= cx + 1; xx++) {
    if (xx < 0 || xx >= 64) continue;
    u8g2.drawLine(xx, max(32, cy - r), xx, min(127, cy + r));
  }

  u8g2.setDrawColor(1);
}

static void draw_white_crosshair(int cx, int cy) {
  const int r = 14;   // longer
  const int t = 2;    // half-thickness, total thickness = 5 pixels

  u8g2.setDrawColor(1);   // white

  // Thick horizontal line
  for (int yy = cy - t; yy <= cy + t; yy++) {
    if (yy < 32 || yy >= 128) continue;
    u8g2.drawLine(max(0, cx - r), yy, min(63, cx + r), yy);
  }

  // Thick vertical line
  for (int xx = cx - t; xx <= cx + t; xx++) {
    if (xx < 0 || xx >= 64) continue;
    u8g2.drawLine(xx, max(32, cy - r), xx, min(127, cy + r));
  }
}

static void draw_prediction_overlay(float p, float dx, float dy, float s, float ms) {
  // Face center in OLED preview coordinates.
  int cx = 32 + (int)roundf(dx * 64.0f);
  int cy = 32 + 48 + (int)roundf(dy * 96.0f);

  if (cx < 0) cx = 0;
  if (cx > 63) cx = 63;
  if (cy < 32) cy = 32;
  if (cy > 127) cy = 127;

  draw_white_crosshair(cx, cy);

  // Top black text panel with white text.
  u8g2.setDrawColor(0);
  u8g2.drawBox(0, 0, 64, 31);
  u8g2.setDrawColor(1);

  char line[24];
  u8g2.setFont(u8g2_font_5x8_tf);

  snprintf(line, sizeof(line), "p%.2f %.0fms", p, ms);
  u8g2.drawStr(0, 8, line);

  snprintf(line, sizeof(line), "dx%+.2f", dx);
  u8g2.drawStr(0, 18, line);

  snprintf(line, sizeof(line), "dy%+.2f", dy);
  u8g2.drawStr(0, 28, line);

  u8g2.sendBuffer();
}

static void draw_running_overlay() {
  u8g2.setDrawColor(0);
  u8g2.drawBox(0, 0, 64, 31);
  u8g2.setDrawColor(1);
  u8g2.setFont(u8g2_font_5x8_tf);
  u8g2.drawStr(0, 12, "RUNNING");
  u8g2.drawStr(0, 24, "NOODLE...");
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

  // Reinitialize I2C for OLED + TCA9555 keys.
  Wire.end();
  delay(50);
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);

  u8g2.begin();
  u8g2.setPowerSave(0);
  oled_message("CAMERA OK", "NOODLE READY");

  init_keys();

  noodle_buffer_init(&X);
  noodle_buffer_init(&A);
  noodle_buffer_init(&B);

  // Pre-allocate the input buffer once. After this, the main loop can
  // copy the 96x96 crop into X without reallocating while the AI task exists.
  noodle_buffer_require(&X, (size_t)IMG_W * IMG_H * IMG_C);

  // Arduino loop normally runs on core 1. Put Noodle inference on core 0.
  xTaskCreatePinnedToCore(
    noodle_ai_task,
    "noodle_ai",
    12288,
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

  // Submit this frame to the AI core only when it is idle.
  // If AI is busy, this frame is still shown as live preview.
  submit_frame_for_ai(fb);

  // Draw camera preview every frame using the latest camera frame.
  draw_camera_crop(fb);

  esp_camera_fb_return(fb);
  fb = nullptr;

  // Copy the latest AI result quickly and safely.
  float p, dx, dy, s, raw_dx, raw_dy, ms;
  bool has_result;
  bool busy;

  portENTER_CRITICAL(&ai_mux);
  p = g_p;
  dx = g_dx_f;
  dy = g_dy_f;
  raw_dx = g_dx;
  raw_dy = g_dy;
  s = g_s;
  ms = g_last_ms;
  has_result = ai_has_result;
  busy = ai_busy;
  portEXIT_CRITICAL(&ai_mux);

  // Draw crosshair every video frame using the latest available prediction.
  // If no prediction is ready yet, just send the live preview.
  if (has_result) {
    draw_prediction_overlay(p, dx, dy, s, ms);
  } else {
    u8g2.sendBuffer();
  }

  // Throttled serial debug only.
  uint32_t now = millis();
  if (now - last_print_ms >= 500) {
    last_print_ms = now;
    Serial.printf(
      "FACE p=%.4f dx=%+.4f dy=%+.4f raw_dx=%+.4f raw_dy=%+.4f s=%.4f t=%.2fms busy=%d\n",
      p, dx, dy, raw_dx, raw_dy, s, ms, busy ? 1 : 0
    );
  }
}
