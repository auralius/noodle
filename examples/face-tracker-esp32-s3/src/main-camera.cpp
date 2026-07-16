#include <Arduino.h>
#include <Wire.h>
#include <U8g2lib.h>
#include "esp_camera.h"

#define I2C_SDA 11
#define I2C_SCL 10

//U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
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

#define TCA_ADDR 0x20

#define TCA_INPUT_PORT0   0x00
#define TCA_INPUT_PORT1   0x01
#define TCA_CONFIG_PORT0  0x06
#define TCA_CONFIG_PORT1  0x07

#define KEY1_BIT 1  // EXIO9  = P11
#define KEY2_BIT 2  // EXIO10 = P12
#define KEY3_BIT 3  // EXIO11 = P13

static uint8_t tca_read_reg(uint8_t reg) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);

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
  // Set P11/P12/P13 as inputs.
  uint8_t cfg1 = tca_read_reg(TCA_CONFIG_PORT1);
  cfg1 |= (1 << KEY1_BIT) | (1 << KEY2_BIT) | (1 << KEY3_BIT);
  tca_write_reg(TCA_CONFIG_PORT1, cfg1);
}

static bool key1_raw_pressed() {
  uint8_t p1 = tca_read_reg(TCA_INPUT_PORT1);

  // Buttons are normally pulled high, pressed = low.
  return ((p1 & (1 << KEY1_BIT)) == 0);
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

static bool init_camera() {
  camera_config_t config;
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

  config.frame_size = FRAMESIZE_QQVGA;  // 160x120
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  return err == ESP_OK;
}

static void oled_message(const char *a, const char *b = nullptr) {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  u8g2.drawStr(0, 14, a);
  if (b) u8g2.drawStr(0, 30, b);
  u8g2.sendBuffer();
}

static void draw_camera_crop(camera_fb_t *fb) {
  // QQVGA grayscale = 160x120
  const int fw = fb->width;
  const int fh = fb->height;

  // Center 96x96 crop
  const int crop = 96;
  const int x0 = (fw - crop) / 2;
  const int y0 = (fh - crop) / 2;

  u8g2.clearBuffer();

  // Draw 64x96 portrait preview
  for (int oy = 0; oy < 96; oy++) {
    int sy = y0 + (oy * crop) / 96;

    for (int ox = 0; ox < 64; ox++) {
      int sx = x0 + (ox * crop) / 64;
      uint8_t g = fb->buf[sy * fw + sx];

      if (g > 110) {
        u8g2.drawPixel(ox, 32 + oy);  // y offset leaves text area on top
      }
    }
  }

  // Text/debug area at top
  u8g2.setFont(u8g2_font_5x8_tf);
  u8g2.drawStr(0, 8,  "CAM OK");
  u8g2.drawStr(0, 18, "96 CROP");

  // Center cross on preview
  // preview starts at y = 32, height = 96
  const int cx = 32;
  const int cy = 32 + 48;

  u8g2.drawLine(cx - 8, cy, cx + 8, cy);
  u8g2.drawLine(cx, cy - 8, cx, cy + 8);

  u8g2.sendBuffer();
}

void setup() {
  Serial.begin(921600);
  delay(1000);

  // 1. Camera first
  if (!init_camera()) {
    // Serial may fail, so just stop here
    while (true) delay(1000);
  }

  // 2. Then re-init I2C for OLED
  Wire.end();
  delay(50);
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);

  u8g2.begin();
  u8g2.setPowerSave(0);

  oled_message("CAMERA OK", "OLED OK");

  init_keys();
}

static void oled_frozen_label() {
  u8g2.setDrawColor(0);
  u8g2.drawBox(0, 20, 64, 10);

  u8g2.setDrawColor(1);
  u8g2.setFont(u8g2_font_5x8_tf);
  u8g2.drawStr(0, 28, "FROZEN");

  u8g2.sendBuffer();
}

void loop() {
  static bool frozen = false;

  // KEY1 toggles freeze/live
  if (key1_pressed_once()) {
    frozen = !frozen;

    if (frozen) {
      oled_frozen_label();
    } else {
      oled_message("LIVE", "preview...");
      delay(200);
    }
  }

  // If frozen, keep the current OLED image.
  // Do not grab new camera frame.
  if (frozen) {
    delay(30);
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) {
    oled_message("Frame FAIL");
    delay(200);
    return;
  }

  draw_camera_crop(fb);

  esp_camera_fb_return(fb);
  delay(100);
}