// camera_pins.h
// Fixed pin map for ESP32-S3 EYE camera (RHYX/OV-class DVP).
// Drop-in header for projects that always use the same board/sensor wiring.

#pragma once
#include <stdint.h>

// CAMERA_MODEL_ESP32S3_EYE
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM  4
#define SIOC_GPIO_NUM  5

#define Y2_GPIO_NUM 11
#define Y3_GPIO_NUM 9
#define Y4_GPIO_NUM 8
#define Y5_GPIO_NUM 10
#define Y6_GPIO_NUM 12
#define Y7_GPIO_NUM 18
#define Y8_GPIO_NUM 17
#define Y9_GPIO_NUM 16

#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM  7
#define PCLK_GPIO_NUM  13


// ------ Convenience: copy pins into camera_config_t ------
#ifdef __cplusplus
  #include <esp_camera.h>
  static inline void cam_fill_pins(camera_config_t& c) {
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer   = LEDC_TIMER_0;
    c.pin_d0       = Y2_GPIO_NUM;
    c.pin_d1       = Y3_GPIO_NUM;
    c.pin_d2       = Y4_GPIO_NUM;
    c.pin_d3       = Y5_GPIO_NUM;
    c.pin_d4       = Y6_GPIO_NUM;
    c.pin_d5       = Y7_GPIO_NUM;
    c.pin_d6       = Y8_GPIO_NUM;
    c.pin_d7       = Y9_GPIO_NUM;
    c.pin_xclk     = XCLK_GPIO_NUM;
    c.pin_pclk     = PCLK_GPIO_NUM;
    c.pin_vsync    = VSYNC_GPIO_NUM;
    c.pin_href     = HREF_GPIO_NUM;
    c.pin_sccb_sda = SIOD_GPIO_NUM;
    c.pin_sccb_scl = SIOC_GPIO_NUM;
    c.pin_pwdn     = PWDN_GPIO_NUM;
    c.pin_reset    = RESET_GPIO_NUM;
    c.xclk_freq_hz = 20000000;
    c.pixel_format = PIXFORMAT_GRAYSCALE; 
    c.grab_mode    = CAMERA_GRAB_LATEST; //CAMERA_GRAB_LATEST
    c.fb_location  = CAMERA_FB_IN_PSRAM;
    c.frame_size   = FRAMESIZE_VGA;
    c.fb_count     = 2;
  }

#endif
