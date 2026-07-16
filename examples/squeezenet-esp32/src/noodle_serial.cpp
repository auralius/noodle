#include "noodle_serial.h"

namespace {

static constexpr size_t CHUNK_SIZE = 64;

}  // namespace

namespace NoodleSerial {

void begin(uint32_t baud) {
  Serial.begin(baud);
  Serial.setTimeout(20);
  delay(1000);
}

void clear_input() {
  while (Serial.available()) Serial.read();
}

void print_ready() {
  Serial.println(F("READY"));
}

bool recv_image_chunked(uint8_t *dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;

  Serial.println(F("RDYIMG"));

  while (got < n) {
    const size_t want = min((size_t)CHUNK_SIZE, n - got);

    size_t this_got = 0;
    uint32_t t0 = millis();

    while (this_got < want) {
      if ((millis() - t0) > timeout_ms) {
        Serial.printf("ERR_TIMEOUT got=%u expected=%u\n",
                      (unsigned)(got + this_got),
                      (unsigned)n);
        return false;
      }

      if (Serial.available() > 0) {
        int c = Serial.read();
        if (c >= 0) {
          dst[got + this_got] = (uint8_t)c;
          this_got++;
        }
      } else {
        delay(0);
      }
    }

    got += want;
    Serial.println(F("ACK"));
  }

  return true;
}

void send_output_image(const uint8_t *buf, size_t n, uint32_t dt_us) {
  Serial.printf("OUT %lu\n", (unsigned long)dt_us);
  Serial.flush();
  delay(2);

  size_t sent = 0;
  while (sent < n) {
    const size_t chunk = min((size_t)64, n - sent);
    size_t w = Serial.write(buf + sent, chunk);
    Serial.flush();

    if (w > 0) {
      sent += w;
    } else {
      delay(1);
    }

    delay(1);  // important for USB CDC stability
  }

  delay(5);
  Serial.print('\n');
  print_ready();
  Serial.flush();
}

bool wait_for_img_header(uint32_t timeout_ms) {
  const uint32_t t0 = millis();
  uint8_t state = 0;

  while ((millis() - t0) <= timeout_ms) {
    if (Serial.available() <= 0) {
      delay(1);
      continue;
    }

    const int c = Serial.read();
    if (c < 0) continue;

    if (state == 0) {
      state = (c == 'I') ? 1 : 0;
    } else if (state == 1) {
      if (c == 'M') state = 2;
      else state = (c == 'I') ? 1 : 0;
    } else {  // state == 2
      if (c == 'G') return true;
      state = (c == 'I') ? 1 : 0;
    }
  }

  return false;
}

}  // namespace NoodleSerial
