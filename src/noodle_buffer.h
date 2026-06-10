/**
 * @file noodle_buffer.h
 * @brief Grow-only float buffers used by NoodleBuffer convolution overloads.
 * @ingroup noodle_public
 */

#ifndef NOODLE_BUFFER_H
#define NOODLE_BUFFER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Grow-only float buffer managed by Noodle.
 * @ingroup noodle_public
 *
 * NoodleBuffer is intentionally simple:
 * - data is either NULL or memory allocated by Noodle.
 * - capacity is expressed in float elements, not bytes.
 * - the buffer grows when required, but never shrinks automatically.
 * - memory is released only when noodle_buffer_free() is called.
 *
 * External/user-owned memory should remain as a raw float pointer and should not
 * be stored inside NoodleBuffer.
 */
typedef struct {
  float *data;
  size_t capacity;
} NoodleBuffer;

/**
 * @brief Initialize a NoodleBuffer.
 * @ingroup noodle_public
 *
 * Sets the data pointer to NULL and the capacity to zero. Call this before the
 * first use of a stack- or static-allocated NoodleBuffer.
 *
 * @param buf Buffer descriptor to initialize. Passing NULL is allowed.
 */
void noodle_buffer_init(NoodleBuffer *buf);

/**
 * @brief Ensure that a buffer can hold at least required_floats floats.
 *
 * If the buffer is NULL, this allocates it.
 * If the buffer is already large enough, this reuses it.
 * If the buffer is too small, this allocates a larger block first, then frees
 * the old block only after the new allocation succeeds.
 *
 * @param buf Buffer descriptor.
 * @param required_floats Required capacity in float elements.
 * @return Pointer to usable float storage, or NULL on failure.
 * @ingroup noodle_public
 */
float *noodle_buffer_require(NoodleBuffer *buf, size_t required_floats);

/**
 * @brief Release a NoodleBuffer.
 * @ingroup noodle_public
 *
 * This frees the internal data pointer and resets the descriptor.
 *
 * @param buf Buffer descriptor to release. Passing NULL is allowed.
 */
void noodle_buffer_free(NoodleBuffer *buf);

/**
 * @brief Return the buffer capacity in float elements.
 * @ingroup noodle_public
 *
 * @param buf Buffer descriptor to inspect.
 * @return Capacity in float elements, or 0 when @p buf is NULL.
 */
size_t noodle_buffer_capacity(const NoodleBuffer *buf);

/**
 * @brief Return the buffer capacity in bytes.
 * @ingroup noodle_public
 *
 * @param buf Buffer descriptor to inspect.
 * @return Capacity in bytes, or 0 when @p buf is NULL.
 */
size_t noodle_buffer_capacity_bytes(const NoodleBuffer *buf);

#ifdef __cplusplus
}
#endif

#endif  // NOODLE_BUFFER_H
