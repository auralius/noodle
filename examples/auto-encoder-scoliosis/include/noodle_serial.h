#pragma once

#include <Arduino.h>
#include <stddef.h>
#include <stdint.h>

namespace NoodleSerial {

static constexpr uint32_t BAUD = 921600;
static constexpr uint32_t RX_TIMEOUT_MS = 1000;

void begin(uint32_t baud = BAUD);
void clear_input();
void print_ready();
bool wait_for_img_header(uint32_t timeout_ms = RX_TIMEOUT_MS);
bool recv_image_chunked(uint8_t *dst, size_t n, uint32_t timeout_ms = RX_TIMEOUT_MS);
void send_output_image(const uint8_t *buf, size_t n, uint32_t dt_us);

}  // namespace NoodleSerial
