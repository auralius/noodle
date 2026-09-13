/**
 * @file noodle_buffer.h
 * @brief Transparent grow-only buffers backed by one byte-addressed packed arena.
 * @ingroup noodle_public
 */
#ifndef NOODLE_BUFFER_H
#define NOODLE_BUFFER_H

#include <stddef.h>
#include <stdint.h>
#include "noodle_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NoodleBuffer {
  NoodleData *data;                    ///< Current storage pointer, or NULL.
  size_t capacity;                     ///< Retained logical capacity in elements.
  size_t _arena_offset_bytes;
  struct NoodleBuffer *_arena_prev;
  struct NoodleBuffer *_arena_next;
  uint32_t _arena_cookie;
} NoodleBuffer;

void noodle_buffer_init(NoodleBuffer *buf);
NoodleData *noodle_buffer_require(NoodleBuffer *buf, size_t required_elements);
void noodle_buffer_free(NoodleBuffer *buf);
size_t noodle_buffer_capacity(const NoodleBuffer *buf);
size_t noodle_buffer_capacity_bytes(const NoodleBuffer *buf);
size_t noodle_buffer_arena_capacity(void);
size_t noodle_buffer_arena_used(void);
size_t noodle_buffer_arena_capacity_bytes(void);
size_t noodle_buffer_arena_used_bytes(void);
size_t noodle_buffer_arena_buffer_count(void);
size_t noodle_buffer_arena_realloc_count(void);
size_t noodle_buffer_arena_move_count(void);
size_t noodle_buffer_arena_moved_bytes(void);

#ifdef __cplusplus
}
#endif
#endif
