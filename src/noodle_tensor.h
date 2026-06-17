/**
 * @file noodle_tensor.h
 * @brief Lightweight tensor metadata helpers built on grow-only NoodleBuffer storage.
 * @ingroup noodle_public
 *
 * NoodleTensor is a small metadata layer over NoodleBuffer. It keeps the
 * existing packed tensor layouts used by noodle.h while carrying enough shape
 * information for the tensor-level layer wrappers to size outputs automatically.
 */

#ifndef NOODLE_TENSOR_H
#define NOODLE_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#include "noodle_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Tensor rank marker used by NoodleTensor.
 * @ingroup noodle_public
 */
typedef enum {
  NOODLE_TENSOR_EMPTY = 0,  ///< Tensor has no valid shape.
  NOODLE_TENSOR_1D    = 1,  ///< Packed `[C][W]` tensor.
  NOODLE_TENSOR_2D    = 2   ///< Packed `[C][W][W]` tensor.
} NoodleTensorRank;

/**
 * @brief Grow-only flat tensor with lightweight shape metadata.
 * @ingroup noodle_public
 *
 * NoodleTensor owns a NoodleBuffer for grow-only storage and adds shape
 * metadata on top. Tensor helpers reuse the same allocation policy as
 * NoodleBuffer while tracking the logical packed shape.
 *
 * Rank-2 tensors are packed as `[C][W][W]`. Rank-1 tensors are packed as
 * `[C][W]`; vectors created by noodle_tensor_require_vector() use `[N][1]`.
 */
typedef struct {
  NoodleBuffer buffer;  ///< Grow-only backing storage.
  uint16_t C;           ///< Channel count, or vector length for rank-1 FCN output.
  uint16_t W;           ///< Spatial width/length; 1 for flat vectors.
  uint8_t rank;         ///< One of NoodleTensorRank.
} NoodleTensor;

/**
 * @brief Initialize a NoodleTensor descriptor.
 * @ingroup noodle_public
 *
 * Sets the backing buffer to empty storage and clears the logical shape.
 *
 * @param t Tensor descriptor to initialize. Passing NULL is allowed.
 */
void noodle_tensor_init(NoodleTensor *t);

/**
 * @brief Release tensor storage and clear shape metadata.
 * @ingroup noodle_public
 *
 * Frees the owned NoodleBuffer storage and resets the tensor to
 * NOODLE_TENSOR_EMPTY.
 *
 * @param t Tensor descriptor to release. Passing NULL is allowed.
 */
void noodle_tensor_free(NoodleTensor *t);

/**
 * @brief Ensure storage for a packed rank-1 `[C][W]` tensor.
 * @ingroup noodle_public
 *
 * The backing buffer grows to at least `C * W` floats when needed. On success,
 * the tensor rank and shape are updated to match the requested layout.
 *
 * @param t Tensor descriptor to grow.
 * @param C Channel count, or vector length when @p W is 1.
 * @param W Per-channel vector length.
 * @return Pointer to float storage, or NULL on null tensor, zero shape, or
 * allocation failure.
 */
float *noodle_tensor_require_1d(NoodleTensor *t, uint16_t C, uint16_t W);

/**
 * @brief Ensure storage for a packed rank-2 `[C][W][W]` tensor.
 * @ingroup noodle_public
 *
 * The backing buffer grows to at least `C * W * W` floats when needed. On
 * success, the tensor rank and shape are updated to match the requested layout.
 *
 * @param t Tensor descriptor to grow.
 * @param C Number of channel planes.
 * @param W Plane width and height.
 * @return Pointer to float storage, or NULL on null tensor, zero shape, or
 * allocation failure.
 */
float *noodle_tensor_require_2d(NoodleTensor *t, uint16_t C, uint16_t W);

/**
 * @brief Ensure storage for a flat vector tensor.
 * @ingroup noodle_public
 *
 * This is a convenience wrapper for noodle_tensor_require_1d(@p t, @p N, 1).
 *
 * @param t Tensor descriptor to grow.
 * @param N Number of vector elements.
 * @return Pointer to float storage, or NULL on null tensor, zero length, or
 * allocation failure.
 */
float *noodle_tensor_require_vector(NoodleTensor *t, uint16_t N);

/**
 * @brief Return the logical tensor size in float elements.
 * @ingroup noodle_public
 *
 * @param t Tensor descriptor to inspect.
 * @return `C * W` for rank-1 tensors, `C * W * W` for rank-2 tensors, or 0
 * when @p t is NULL, empty, or has an unknown rank.
 */
size_t noodle_tensor_size(const NoodleTensor *t);

/**
 * @brief Return the backing buffer capacity in float elements.
 * @ingroup noodle_public
 *
 * @param t Tensor descriptor to inspect.
 * @return Backing capacity in float elements, or 0 when @p t is NULL.
 */
size_t noodle_tensor_capacity(const NoodleTensor *t);

/**
 * @brief Return the backing buffer capacity in bytes.
 * @ingroup noodle_public
 *
 * @param t Tensor descriptor to inspect.
 * @return Backing capacity in bytes, or 0 when @p t is NULL.
 */
size_t noodle_tensor_capacity_bytes(const NoodleTensor *t);

/**
 * @brief Return mutable tensor data.
 * @ingroup noodle_public
 *
 * @param t Tensor descriptor to inspect.
 * @return Backing float pointer, or NULL when @p t is NULL or unallocated.
 */
static inline float *noodle_tensor_data(NoodleTensor *t) {
  return t ? t->buffer.data : NULL;
}

/**
 * @brief Return immutable tensor data.
 * @ingroup noodle_public
 *
 * @param t Tensor descriptor to inspect.
 * @return Backing float pointer, or NULL when @p t is NULL or unallocated.
 */
static inline const float *noodle_tensor_const_data(const NoodleTensor *t) {
  return t ? t->buffer.data : NULL;
}

/**
 * @brief Return the mutable backing NoodleBuffer.
 * @ingroup noodle_public
 *
 * @param t Tensor descriptor to inspect.
 * @return Backing buffer descriptor, or NULL when @p t is NULL.
 */
static inline NoodleBuffer *noodle_tensor_buffer(NoodleTensor *t) {
  return t ? &t->buffer : NULL;
}

/**
 * @brief Return the immutable backing NoodleBuffer.
 * @ingroup noodle_public
 *
 * @param t Tensor descriptor to inspect.
 * @return Backing buffer descriptor, or NULL when @p t is NULL.
 */
static inline const NoodleBuffer *noodle_tensor_const_buffer(const NoodleTensor *t) {
  return t ? &t->buffer : NULL;
}

/**
 * @brief Compute the flat index for packed `[C][W]` tensor data.
 * @ingroup noodle_public
 *
 * This helper performs no bounds checks.
 *
 * @param c Channel index.
 * @param x Position within the channel.
 * @param W Per-channel vector length.
 * @return Flat element index `c * W + x`.
 */
static inline size_t noodle_tensor_idx1d(uint16_t c, uint16_t x, uint16_t W) {
  return (size_t)c * (size_t)W + (size_t)x;
}

/**
 * @brief Compute the flat index for packed `[C][W][W]` tensor data.
 * @ingroup noodle_public
 *
 * This helper performs no bounds checks.
 *
 * @param c Channel index.
 * @param y Row index within the channel plane.
 * @param x Column index within the channel plane.
 * @param W Plane width and height.
 * @return Flat element index `((c * W) + y) * W + x`.
 */
static inline size_t noodle_tensor_idx2d(uint16_t c, uint16_t y, uint16_t x, uint16_t W) {
  return ((size_t)c * (size_t)W + (size_t)y) * (size_t)W + (size_t)x;
}

#ifdef __cplusplus
}
#endif

#endif  // NOODLE_TENSOR_H
