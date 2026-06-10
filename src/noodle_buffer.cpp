/**
 * @file noodle_buffer.cpp
 * @brief Grow-only NoodleBuffer allocation helpers.
 * @ingroup noodle_api
 */

#include "noodle_buffer.h"

#include <stdlib.h>

#if defined(ARDUINO_ARCH_ESP32)
#include <Arduino.h>  // psramFound(), ps_malloc()
#endif

static float *noodle_buffer_alloc_float(size_t n_floats) {
  if (n_floats == 0) return NULL;

  const size_t bytes = n_floats * sizeof(float);

  /*
   * Workspace/scratch buffers are usually accessed repeatedly in inner loops,
   * so internal RAM is preferred. On ESP32, PSRAM is used only as fallback.
   */
  float *p = (float *)malloc(bytes);

#if defined(ARDUINO_ARCH_ESP32)
  if (!p && psramFound()) {
    p = (float *)ps_malloc(bytes);
  }
#endif

  return p;
}

void noodle_buffer_init(NoodleBuffer *buf) {
  if (!buf) return;

  buf->data = NULL;
  buf->capacity = 0;
}

float *noodle_buffer_require(NoodleBuffer *buf, size_t required_floats) {
  if (!buf || required_floats == 0) return NULL;

  if (buf->data && buf->capacity >= required_floats) {
    return buf->data;
  }

  /*
   * Allocate the larger block before freeing the old one.
   * If allocation fails, the old buffer remains valid.
   */
  float *new_data = noodle_buffer_alloc_float(required_floats);
  if (!new_data) {
    return NULL;
  }

  if (buf->data) {
    free(buf->data);
  }

  buf->data = new_data;
  buf->capacity = required_floats;

  return buf->data;
}

void noodle_buffer_free(NoodleBuffer *buf) {
  if (!buf) return;

  if (buf->data) {
    free(buf->data);
  }

  buf->data = NULL;
  buf->capacity = 0;
}

size_t noodle_buffer_capacity(const NoodleBuffer *buf) {
  if (!buf) return 0;
  return buf->capacity;
}

size_t noodle_buffer_capacity_bytes(const NoodleBuffer *buf) {
  if (!buf) return 0;
  return buf->capacity * sizeof(float);
}
