/**
 * @file noodle_int8.h
 * @brief Internal and exporter-facing helpers for NOODLE_USE_INT8 builds.
 */
#pragma once

#include "noodle.h"

#if defined(NOODLE_USE_INT8)

#ifndef NOODLE_SAME_PADDING
#define NOODLE_SAME_PADDING 65535u
#endif

enum NoodleI8PoolMode : uint8_t {
  NOODLE_I8_POOL_MAX = 1,
  NOODLE_I8_POOL_MEAN = 2
};

struct NoodleI8Pool {
  uint16_t M = 1;
  uint16_t T = 1;
  NoodleI8PoolMode mode = NOODLE_I8_POOL_MAX;
};

struct NoodleI8Requant {
  int32_t multiplier = 0;
  int32_t shift = 0;
  float output_scale = 1.0f;
  int32_t output_zero_point = 0;
  int32_t activation_min = -128;
  int32_t activation_max = 127;
};

struct NoodleI8SoftmaxLUT {
  const uint16_t *exp_q15 = nullptr;
  float input_scale = 1.0f;
};

// Tensor allocation helpers that also attach quantization metadata.
NoodleData *noodle_i8_tensor_require_1d(NoodleTensor *t, uint16_t C, uint16_t W,
                                        float scale, int32_t zero_point);
NoodleData *noodle_i8_tensor_require_2d(NoodleTensor *t, uint16_t C, uint16_t W,
                                        float scale, int32_t zero_point);
NoodleData *noodle_i8_tensor_require_vector(NoodleTensor *t, uint16_t N,
                                            float scale, int32_t zero_point);

// Integer arithmetic and boundary conversion helpers.
int8_t noodle_i8_clamp(int32_t x);
int32_t noodle_i8_saturating_rounding_doubling_high_mul(int32_t a, int32_t b);
int32_t noodle_i8_rounding_divide_by_pot(int32_t x, int exponent);
int32_t noodle_i8_multiply_by_quantized_multiplier(int32_t x,
                                                   int32_t multiplier,
                                                   int32_t shift);
bool noodle_quantize_multiplier(double real_multiplier,
                                int32_t *multiplier,
                                int32_t *shift);
int32_t noodle_quantize_float(float value, float scale, int32_t zero_point);
float noodle_dequantize_int8(int8_t value, float scale, int32_t zero_point);
void noodle_quantize_array(const float *src, int8_t *dst, size_t n,
                           float scale, int32_t zero_point);
void noodle_dequantize_array(const int8_t *src, float *dst, size_t n,
                             float scale, int32_t zero_point);
int32_t noodle_activation_min_relu(float output_scale,
                                   int32_t output_zero_point);
int32_t noodle_read_i32(NDL_File &f);
void noodle_write_i32(NDL_File &f, int32_t value);

// Internal full-int8 layer kernels. Public wrappers keep the old Noodle API.
uint16_t noodle_i8_conv2d(NoodleTensor *input, const Conv &layer,
                          NoodleTensor *output);
uint16_t noodle_i8_conv2d(NoodleTensor *input, const ConvMem &layer,
                          NoodleTensor *output);
uint16_t noodle_i8_conv2d(NoodleTensor *input, const ConvProgmem &layer,
                          NoodleTensor *output);
uint16_t noodle_i8_conv1d(NoodleTensor *input, const Conv &layer,
                          NoodleTensor *output);
uint16_t noodle_i8_conv1d(NoodleTensor *input, const ConvMem &layer,
                          NoodleTensor *output);
uint16_t noodle_i8_dwconv2d(NoodleTensor *input, const Conv &layer,
                            NoodleTensor *output);
uint16_t noodle_i8_dwconv2d(NoodleTensor *input, const ConvMem &layer,
                            NoodleTensor *output);
uint16_t noodle_i8_dwconv2d(NoodleTensor *input, const ConvProgmem &layer,
                            NoodleTensor *output);
uint16_t noodle_i8_conv_transpose2d(NoodleTensor *input, const Conv &layer,
                                    NoodleTensor *output);
uint16_t noodle_i8_conv_transpose2d(NoodleTensor *input, const ConvMem &layer,
                                    NoodleTensor *output);
uint16_t noodle_i8_fcn(NoodleTensor *input, const FCNFile &layer,
                       NoodleTensor *output);
uint16_t noodle_i8_fcn(NoodleTensor *input, const FCNMem &layer,
                       NoodleTensor *output);
uint16_t noodle_i8_fcn(NoodleTensor *input, const FCNProgmem &layer,
                       NoodleTensor *output);
uint16_t noodle_i8_pool2d(NoodleTensor *input, const NoodleI8Pool &pool,
                          NoodleTensor *output);
uint16_t noodle_i8_pool1d(NoodleTensor *input, const NoodleI8Pool &pool,
                          NoodleTensor *output);
uint16_t noodle_i8_gap(NoodleTensor *input, const NoodleI8Requant &requant,
                       NoodleTensor *output);
uint16_t noodle_i8_gmp(NoodleTensor *input, NoodleTensor *output);
uint16_t noodle_i8_flat(NoodleTensor *input, NoodleTensor *output);
uint16_t noodle_i8_reshape_hwc_to_chw(NoodleTensor *input,
                                      uint16_t W, uint16_t C,
                                      NoodleTensor *output);
uint16_t noodle_i8_concat(NoodleTensor *A, NoodleTensor *B,
                          NoodleTensor *output);
uint16_t noodle_i8_relu(NoodleTensor *input_output);
void noodle_i8_find_max(const NoodleTensor *input, uint16_t *index,
                        int8_t *value);
uint16_t noodle_i8_lut(NoodleTensor *input, const int8_t lut[256],
                       float output_scale, int32_t output_zero_point,
                       NoodleTensor *output);
uint16_t noodle_i8_softmax(NoodleTensor *input,
                           const NoodleI8SoftmaxLUT &lut,
                           NoodleTensor *output);
void noodle_build_softmax_lut(float input_scale, uint16_t exp_q15[256]);
void noodle_build_sigmoid_lut(float input_scale, int32_t input_zero_point,
                              int8_t output_lut[256]);
void noodle_build_tanh_lut(float input_scale, int32_t input_zero_point,
                           int8_t output_lut[256]);
void noodle_i8_temp_buffers_free(void);

#endif
