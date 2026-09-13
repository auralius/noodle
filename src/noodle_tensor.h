/**
 * @file noodle_tensor.h
 * @brief Lightweight tensor metadata over grow-only NoodleBuffer storage.
 */
#ifndef NOODLE_TENSOR_H
#define NOODLE_TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include "noodle_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NOODLE_TENSOR_EMPTY = 0,
  NOODLE_TENSOR_1D = 1,
  NOODLE_TENSOR_2D = 2
} NoodleTensorRank;

typedef struct {
  NoodleBuffer buffer;
  uint16_t C;
  uint16_t W;
  uint8_t rank;
#if defined(NOODLE_USE_INT8)
  float scale;          ///< real = (q - zero_point) * scale
  int32_t zero_point;
#endif
} NoodleTensor;

void noodle_tensor_init(NoodleTensor *t);
void noodle_tensor_free(NoodleTensor *t);
NoodleData *noodle_tensor_require_1d(NoodleTensor *t, uint16_t C, uint16_t W);
NoodleData *noodle_tensor_require_2d(NoodleTensor *t, uint16_t C, uint16_t W);
NoodleData *noodle_tensor_require_vector(NoodleTensor *t, uint16_t N);
size_t noodle_tensor_size(const NoodleTensor *t);
size_t noodle_tensor_capacity(const NoodleTensor *t);
size_t noodle_tensor_capacity_bytes(const NoodleTensor *t);

#if defined(NOODLE_USE_INT8)
void noodle_tensor_set_quantization(NoodleTensor *t, float scale, int32_t zero_point);
#endif

static inline NoodleData *noodle_tensor_data(NoodleTensor *t) {
  return t ? t->buffer.data : NULL;
}
static inline const NoodleData *noodle_tensor_const_data(const NoodleTensor *t) {
  return t ? t->buffer.data : NULL;
}
static inline NoodleBuffer *noodle_tensor_buffer(NoodleTensor *t) {
  return t ? &t->buffer : NULL;
}
static inline const NoodleBuffer *noodle_tensor_const_buffer(const NoodleTensor *t) {
  return t ? &t->buffer : NULL;
}
static inline size_t noodle_tensor_idx1d(uint16_t c, uint16_t x, uint16_t W) {
  return (size_t)c * W + x;
}
static inline size_t noodle_tensor_idx2d(uint16_t c, uint16_t y, uint16_t x, uint16_t W) {
  return ((size_t)c * W + y) * W + x;
}

#ifdef __cplusplus
}
#endif
#endif
