/**
 * @file noodle_internal.h
 * @brief Private declarations shared by Noodle implementation files.
 */
#pragma once

#include "noodle.h"
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(NOODLE_USE_SDFAT)
extern SdFat NOODLE_FS;
#endif

extern NDL_File fw;
extern NDL_File fb;
extern NDL_File fo;
extern NDL_File fi;

extern void *temp_buff1;
extern void *temp_buff2;

float *noodle_slice(float *flat, size_t W, size_t z);
size_t noodle_read_raw(NDL_File &f, void *dst, size_t n);
size_t noodle_write_raw(NDL_File &f, const void *src, size_t n);
size_t noodle_read_float_block(NDL_File &f, float *dst, size_t n_floats);
float noodle_dot_float_block(const float *x, const float *w, uint16_t n);

// Private convolution/math helpers
uint16_t noodle_do_conv1d(float *input, float *kernel, uint16_t W, uint16_t K, float *output, uint16_t P, uint16_t S);
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S, NDL_File &fo);
uint16_t noodle_do_pooling1d(const float *input, uint16_t W, uint16_t K, uint16_t S, float *output);
float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P0, int16_t P1);
float noodle_get_padded_x(float *grid, int16_t i, int16_t j, int16_t W, int16_t P0, int16_t P1);
uint16_t noodle_do_bias(float *output, float bias, uint16_t n);
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, NDL_File &fo);
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, float *output);
uint16_t noodle_do_conv(byte *grid, const float *kernel, uint16_t K, uint16_t W, float *output, uint16_t P, uint16_t S);
uint16_t noodle_do_conv(float *grid, const float *kernel, uint16_t K, uint16_t W, float *output, uint16_t P, uint16_t S);
void noodle_reset_buffer(float *buffer, uint16_t n);
uint16_t noodle_do_bias_act(float *output, float bias, uint16_t n, Activation act);
uint16_t noodle_do_conv_transpose(float *input, const float *kernel, uint16_t K, uint16_t W, float *output, uint16_t P, uint16_t S, uint16_t OP);
void noodle_find_max(float *input, uint16_t n, float &max_val, uint16_t &max_idx);
void noodle_unpack_bn_params(const float *bn_params, uint16_t N, const float **gamma, const float **beta, const float **mean, const float **var);
uint16_t noodle_bn1d(float *x, uint16_t N, const float *gamma, const float *beta, const float *mean, const float *var, float eps);
uint16_t noodle_bn1d(float *x, uint16_t N, const float *bn_params, float eps);
uint16_t noodle_bn1d_relu(float *x, uint16_t N, const float *gamma, const float *beta, const float *mean, const float *var, float eps);
uint16_t noodle_bn1d_relu(float *x, uint16_t N, const float *bn_params, float eps);
uint16_t noodle_bn2d(float *x, uint16_t C, uint16_t W, const float *gamma, const float *beta, const float *mean, const float *var, float eps);
uint16_t noodle_bn2d(float *x, uint16_t C, uint16_t W, const float *bn_params, float eps);
uint16_t noodle_bn2d_relu(float *x, uint16_t C, uint16_t W, const float *gamma, const float *beta, const float *mean, const float *var, float eps);
uint16_t noodle_bn2d_relu(float *x, uint16_t C, uint16_t W, const float *bn_params, float eps);
uint16_t noodle_compute_V(uint16_t K, uint16_t W, uint16_t P, uint16_t S);
uint16_t noodle_compute_V_and_P(uint16_t K, uint16_t W, uint16_t P, uint16_t S, uint16_t &P0, uint16_t &P1);
uint16_t noodle_valid_max_pool(float *inplace, uint16_t W, uint16_t C, const Pool &pool);
uint16_t noodle_compute_Vt(uint16_t K, uint16_t W, uint16_t P, uint16_t S, uint16_t OP);
uint16_t noodle_compute_Vt_and_P(uint16_t K, uint16_t W, uint16_t P, uint16_t S, uint16_t OP, uint16_t &P0, uint16_t &P1);
