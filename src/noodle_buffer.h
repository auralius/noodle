/**
 * @file noodle_buffer.h
 * @brief Transparent grow-only float buffers backed by one byte-addressed packed arena.
 * @ingroup noodle_public
 */

#ifndef NOODLE_BUFFER_H
#define NOODLE_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Grow-only float buffer managed transparently by Noodle.
 * @ingroup noodle_public
 *
 * All initialized NoodleBuffer objects are movable slices of one hidden global
 * byte-addressed arena. The public API remains float-oriented for compatibility:
 * capacity and noodle_buffer_require() use float elements. Arena placement is
 * tracked internally in bytes so the allocator can later support other element
 * types and alignment rules.
 *
 * @warning A raw pointer copied from data can become stale after any later call
 * to noodle_buffer_require(), because the arena or another logical buffer may
 * move. Read buf->data again after sizing operations.
 */
typedef struct NoodleBuffer {
  float *data;                         ///< Current float storage pointer, or NULL.
  size_t capacity;                    ///< Retained logical capacity in floats.

  size_t _arena_offset_bytes;         ///< Internal byte offset in the global arena.
  struct NoodleBuffer *_arena_prev;   ///< Internal packed-order predecessor.
  struct NoodleBuffer *_arena_next;   ///< Internal packed-order successor.
  uint32_t _arena_cookie;             ///< Internal registration marker.
} NoodleBuffer;

/** @brief Initialize and register a NoodleBuffer. */
void noodle_buffer_init(NoodleBuffer *buf);

/**
 * @brief Ensure that a buffer can hold at least required_floats floats.
 * @return Pointer to usable float storage, or NULL on failure.
 */
float *noodle_buffer_require(NoodleBuffer *buf, size_t required_floats);

/** @brief Release and unregister a NoodleBuffer. */
void noodle_buffer_free(NoodleBuffer *buf);

/** @brief Return the buffer capacity in float elements. */
size_t noodle_buffer_capacity(const NoodleBuffer *buf);

/** @brief Return the buffer capacity in bytes. */
size_t noodle_buffer_capacity_bytes(const NoodleBuffer *buf);

/**
 * @name Optional hidden-arena diagnostics
 * @{ */

/**
 * @brief Return physical arena capacity as an equivalent number of floats.
 * @note Prefer noodle_buffer_arena_capacity_bytes() for new code.
 */
size_t noodle_buffer_arena_capacity(void);

/**
 * @brief Return packed arena usage as an equivalent number of floats.
 * @note Prefer noodle_buffer_arena_used_bytes() for new code.
 */
size_t noodle_buffer_arena_used(void);

/** @brief Return the physical global-arena capacity in bytes. */
size_t noodle_buffer_arena_capacity_bytes(void);

/** @brief Return the packed logical arena usage in bytes. */
size_t noodle_buffer_arena_used_bytes(void);

/** @brief Return the number of currently registered NoodleBuffer objects. */
size_t noodle_buffer_arena_buffer_count(void);

/** @brief Return the number of successful physical arena allocations/resizes. */
size_t noodle_buffer_arena_realloc_count(void);

/** @brief Return the number of packed suffix moves. */
size_t noodle_buffer_arena_move_count(void);

/** @brief Return the cumulative bytes moved while compacting packed suffixes. */
size_t noodle_buffer_arena_moved_bytes(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  // NOODLE_BUFFER_H
