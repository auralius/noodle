/**
 * @file noodle_buffer.cpp
 * @brief Transparent byte-addressed packed global arena for NoodleBuffer.
 * @ingroup noodle_api
 */

#include "noodle_buffer.h"
#include "noodle_config.h"

#include <stdlib.h>
#include <string.h>

#if defined(ARDUINO_ARCH_ESP32)
#include <Arduino.h>
#include <esp_heap_caps.h>
#endif

#ifndef NOODLE_BUFFER_ARENA_INITIAL_BYTES
#define NOODLE_BUFFER_ARENA_INITIAL_BYTES 64u
#endif

#define NOODLE_BUFFER_ARENA_COOKIE 0x4E424146u /* "NBAF" */

typedef struct {
  uint8_t *data;
  size_t capacity_bytes;
  size_t used_bytes;
  size_t buffer_count;

  NoodleBuffer *first;
  NoodleBuffer *last;

  size_t realloc_count;
  size_t move_count;
  size_t moved_bytes;
} NoodleBufferGlobalArena;

static NoodleBufferGlobalArena g_noodle_buffer_arena = {
  NULL, 0, 0, 0, NULL, NULL, 0, 0, 0
};

static int noodle_buffer_element_bytes(size_t n_elements, size_t *bytes_out) {
  if (!bytes_out) return 0;
  if (n_elements > ((size_t)-1) / sizeof(NoodleData)) return 0;
  *bytes_out = n_elements * sizeof(NoodleData);
  return 1;
}

static NoodleData *noodle_buffer_pointer_at(size_t offset_bytes) {
  if (!g_noodle_buffer_arena.data) return NULL;
  return (NoodleData *)(void *)(g_noodle_buffer_arena.data + offset_bytes);
}

static void noodle_buffer_refresh_all_pointers(void) {
  NoodleBuffer *b = g_noodle_buffer_arena.first;
  while (b) {
    b->data = (b->capacity > 0)
            ? noodle_buffer_pointer_at(b->_arena_offset_bytes)
            : NULL;
    b = b->_arena_next;
  }
}

static int noodle_buffer_arena_resize_bytes(size_t requested_bytes) {
  if (requested_bytes <= g_noodle_buffer_arena.capacity_bytes) return 1;

  size_t target_bytes = requested_bytes;
  if (target_bytes < (size_t)NOODLE_BUFFER_ARENA_INITIAL_BYTES) {
    target_bytes = (size_t)NOODLE_BUFFER_ARENA_INITIAL_BYTES;
  }

  uint8_t *new_data =
      (uint8_t *)realloc(g_noodle_buffer_arena.data, target_bytes);

#if defined(ARDUINO_ARCH_ESP32)
  if (!new_data && psramFound()) {
    new_data = (uint8_t *)heap_caps_realloc(
        g_noodle_buffer_arena.data,
        target_bytes,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
#endif

  if (!new_data) return 0;

  g_noodle_buffer_arena.data = new_data;
  g_noodle_buffer_arena.capacity_bytes = target_bytes;
  g_noodle_buffer_arena.realloc_count++;
  noodle_buffer_refresh_all_pointers();
  return 1;
}

static void noodle_buffer_arena_ensure_started(void) {
  if (g_noodle_buffer_arena.data ||
      NOODLE_BUFFER_ARENA_INITIAL_BYTES == 0) {
    return;
  }

  (void)noodle_buffer_arena_resize_bytes(
      (size_t)NOODLE_BUFFER_ARENA_INITIAL_BYTES);
}

static void noodle_buffer_register(NoodleBuffer *buf) {
  if (!buf) return;

  buf->data = NULL;
  buf->capacity = 0;
  buf->_arena_offset_bytes = g_noodle_buffer_arena.used_bytes;
  buf->_arena_prev = g_noodle_buffer_arena.last;
  buf->_arena_next = NULL;
  buf->_arena_cookie = NOODLE_BUFFER_ARENA_COOKIE;

  if (g_noodle_buffer_arena.last) {
    g_noodle_buffer_arena.last->_arena_next = buf;
  } else {
    g_noodle_buffer_arena.first = buf;
  }

  g_noodle_buffer_arena.last = buf;
  g_noodle_buffer_arena.buffer_count++;
}

static int noodle_buffer_is_registered(const NoodleBuffer *buf) {
  if (!buf) return 0;

  const NoodleBuffer *b = g_noodle_buffer_arena.first;
  while (b) {
    if (b == buf) return 1;
    b = b->_arena_next;
  }
  return 0;
}

static void noodle_buffer_update_following_offsets(NoodleBuffer *first,
                                                   size_t delta_bytes,
                                                   int move_right) {
  NoodleBuffer *b = first;
  while (b) {
    if (move_right) {
      b->_arena_offset_bytes += delta_bytes;
    } else {
      b->_arena_offset_bytes -= delta_bytes;
    }

    b->data = (b->capacity > 0)
            ? noodle_buffer_pointer_at(b->_arena_offset_bytes)
            : NULL;
    b = b->_arena_next;
  }
}

void noodle_buffer_init(NoodleBuffer *buf) {
  if (!buf) return;

  if (noodle_buffer_is_registered(buf)) {
    noodle_buffer_free(buf);
  }

  noodle_buffer_arena_ensure_started();
  noodle_buffer_register(buf);
}

NoodleData *noodle_buffer_require(NoodleBuffer *buf, size_t required_elements) {
  if (!buf || required_elements == 0) return NULL;

  if (!noodle_buffer_is_registered(buf)) {
    noodle_buffer_arena_ensure_started();
    noodle_buffer_register(buf);
  }

  if (buf->capacity >= required_elements) {
    return buf->data;
  }

  size_t required_buffer_bytes = 0;
  size_t current_buffer_bytes = 0;
  if (!noodle_buffer_element_bytes(required_elements, &required_buffer_bytes) ||
      !noodle_buffer_element_bytes(buf->capacity, &current_buffer_bytes)) {
    return NULL;
  }

  const size_t delta_bytes = required_buffer_bytes - current_buffer_bytes;
  if (delta_bytes > ((size_t)-1) - g_noodle_buffer_arena.used_bytes) {
    return NULL;
  }

  const size_t required_total_bytes =
      g_noodle_buffer_arena.used_bytes + delta_bytes;
  if (!noodle_buffer_arena_resize_bytes(required_total_bytes)) {
    return NULL;
  }

  const size_t suffix_start_bytes =
      buf->_arena_offset_bytes + current_buffer_bytes;
  const size_t suffix_size_bytes =
      g_noodle_buffer_arena.used_bytes - suffix_start_bytes;

  if (suffix_size_bytes > 0) {
    memmove(g_noodle_buffer_arena.data + suffix_start_bytes + delta_bytes,
            g_noodle_buffer_arena.data + suffix_start_bytes,
            suffix_size_bytes);

    g_noodle_buffer_arena.move_count++;
    g_noodle_buffer_arena.moved_bytes += suffix_size_bytes;
  }

  noodle_buffer_update_following_offsets(
      buf->_arena_next, delta_bytes, 1);

  buf->capacity = required_elements;
  buf->data = noodle_buffer_pointer_at(buf->_arena_offset_bytes);
  g_noodle_buffer_arena.used_bytes = required_total_bytes;

  return buf->data;
}

void noodle_buffer_free(NoodleBuffer *buf) {
  if (!buf) return;

  if (!noodle_buffer_is_registered(buf)) {
    buf->data = NULL;
    buf->capacity = 0;
    return;
  }

  size_t released_bytes = 0;
  if (!noodle_buffer_element_bytes(buf->capacity, &released_bytes)) {
    return;
  }

  const size_t suffix_start_bytes =
      buf->_arena_offset_bytes + released_bytes;
  const size_t suffix_size_bytes =
      g_noodle_buffer_arena.used_bytes - suffix_start_bytes;

  if (released_bytes > 0 && suffix_size_bytes > 0) {
    memmove(g_noodle_buffer_arena.data + buf->_arena_offset_bytes,
            g_noodle_buffer_arena.data + suffix_start_bytes,
            suffix_size_bytes);

    g_noodle_buffer_arena.move_count++;
    g_noodle_buffer_arena.moved_bytes += suffix_size_bytes;
  }

  if (released_bytes > 0) {
    noodle_buffer_update_following_offsets(
        buf->_arena_next, released_bytes, 0);
    g_noodle_buffer_arena.used_bytes -= released_bytes;
  }

  if (buf->_arena_prev) {
    buf->_arena_prev->_arena_next = buf->_arena_next;
  } else {
    g_noodle_buffer_arena.first = buf->_arena_next;
  }

  if (buf->_arena_next) {
    buf->_arena_next->_arena_prev = buf->_arena_prev;
  } else {
    g_noodle_buffer_arena.last = buf->_arena_prev;
  }

  if (g_noodle_buffer_arena.buffer_count > 0) {
    g_noodle_buffer_arena.buffer_count--;
  }

  buf->data = NULL;
  buf->capacity = 0;
  buf->_arena_offset_bytes = 0;
  buf->_arena_prev = NULL;
  buf->_arena_next = NULL;
  buf->_arena_cookie = 0;

  if (g_noodle_buffer_arena.buffer_count == 0) {
    free(g_noodle_buffer_arena.data);
    g_noodle_buffer_arena.data = NULL;
    g_noodle_buffer_arena.capacity_bytes = 0;
    g_noodle_buffer_arena.used_bytes = 0;
    g_noodle_buffer_arena.first = NULL;
    g_noodle_buffer_arena.last = NULL;
  }
}

size_t noodle_buffer_capacity(const NoodleBuffer *buf) {
  return buf ? buf->capacity : 0;
}

size_t noodle_buffer_capacity_bytes(const NoodleBuffer *buf) {
  size_t bytes = 0;
  if (!buf || !noodle_buffer_element_bytes(buf->capacity, &bytes)) return 0;
  return bytes;
}

size_t noodle_buffer_arena_capacity(void) {
  return g_noodle_buffer_arena.capacity_bytes / sizeof(NoodleData);
}

size_t noodle_buffer_arena_used(void) {
  return g_noodle_buffer_arena.used_bytes / sizeof(NoodleData);
}

size_t noodle_buffer_arena_capacity_bytes(void) {
  return g_noodle_buffer_arena.capacity_bytes;
}

size_t noodle_buffer_arena_used_bytes(void) {
  return g_noodle_buffer_arena.used_bytes;
}

size_t noodle_buffer_arena_buffer_count(void) {
  return g_noodle_buffer_arena.buffer_count;
}

size_t noodle_buffer_arena_realloc_count(void) {
  return g_noodle_buffer_arena.realloc_count;
}

size_t noodle_buffer_arena_move_count(void) {
  return g_noodle_buffer_arena.move_count;
}

size_t noodle_buffer_arena_moved_bytes(void) {
  return g_noodle_buffer_arena.moved_bytes;
}
