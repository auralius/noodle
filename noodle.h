/**
 * @file noodle.h
 * @brief File-streamed CNN primitives for microcontrollers (backend-agnostic FS).
 *
 * Backend selection is handled by @ref noodle_fs.h. Define one of:
 *   NOODLE_USE_SDFAT, NOODLE_USE_SD_MMC, NOODLE_USE_FFAT
 */
#pragma once

#include <stdint.h>
#ifdef ARDUINO
#include <Arduino.h>
#endif
#include "noodle_fs.h"

#ifndef ARDUINO
typedef unsigned char byte;
#endif

typedef void (*CBFPtr)(float progress);

// Utilities
float* noodle_slice(float* flat, size_t W, size_t z);
void   noodle_setup_temp_buffers(void *b1, void *b2);

// FS helpers
NDL_File noodle_open_file_for_write(const char* fn);
size_t   noodle_read_bytes_until(NDL_File &file, char terminator, char *buffer, size_t length);

// SD init (pins variant meaningful for SD_MMC; others ignore pins)
bool noodle_sd_init(int clk_pin, int cmd_pin, int d0_pin);
bool noodle_sd_init();

void noodle_n2ll(uint16_t number, char *out);

// Scalar I/O helpers
void    noodle_write_float(NDL_File &f, float d);
float   noodle_read_float(NDL_File &f);
byte    noodle_read_byte(NDL_File &f);
void    noodle_write_byte(NDL_File &f, byte d);
void    noodle_delete_file(const char *fn);

// Memory utils
float *noodle_create_buffer(uint16_t size);
void   noodle_delete_buffer(float *buffer);
void   noodle_reset_buffer(float *buffer, uint16_t n);

// Array/Grid I/O
void noodle_array_to_file(float *array, const char *fn, uint16_t n);
void noodle_grid_to_file(byte *grid, const char *fn, uint16_t n);
void noodle_grid_to_file(float *grid, const char *fn, uint16_t n);

float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P);
float noodle_get_padded_x(float *grid, int16_t i, int16_t j, int16_t W, int16_t P);

void noodle_array_from_file(const char *fn, float *buffer, uint16_t K);
void noodle_grid_from_file(const char *fn, byte *buffer, uint16_t K, bool transposed=false);
void noodle_grid_from_file(const char *fn, int8_t *buffer, uint16_t K, bool transposed=false);
void noodle_grid_from_file(const char *fn, float *buffer, uint16_t K, bool transposed=false);

// 2D Convolution
uint16_t noodle_do_conv(byte *grid, float *kernel, uint16_t K, uint16_t W,
                        float *output_buffer, uint16_t P, uint16_t S);
/** @overload noodle_do_conv */
uint16_t noodle_do_conv(float *grid, float *kernel, uint16_t K, uint16_t W,
                        float *output_buffer, uint16_t P, uint16_t S);

uint16_t noodle_conv_byte(const char *in_fn,
                          uint16_t n_inputs,
                          uint16_t n_outputs,
                          const char *out_fn,
                          const char *weight_fn,
                          const char *bias_fn,
                          uint16_t W,
                          uint16_t P,
                          uint16_t K,
                          uint16_t S,
                          uint16_t M,
                          uint16_t T,
                          CBFPtr progress_cb = NULL);

/** @overload noodle_conv_float */
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           const char *weight_fn,
                           const char *bias_fn,
                           uint16_t W,
                           uint16_t P,
                           uint16_t K,
                           uint16_t S,
                           uint16_t M,
                           uint16_t T,
                           CBFPtr progress_cb = NULL);
/** @overload noodle_conv_float */
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           const char *weight_fn,
                           const char *bias_fn,
                           uint16_t W,
                           uint16_t P,
                           uint16_t K,
                           uint16_t S,
                           uint16_t M,
                           uint16_t T,
                           CBFPtr progress_cb = NULL);
/** @overload noodle_conv_float */
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           const char *weight_fn,
                           const char *bias_fn,
                           uint16_t W,
                           uint16_t P,
                           uint16_t K,
                           uint16_t S,
                           uint16_t M,
                           uint16_t T,
                           CBFPtr progress_cb = NULL);
/** @overload noodle_conv_float */
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           const char *weight_fn,
                           const char *bias_fn,
                           uint16_t W,
                           uint16_t P,
                           uint16_t K,
                           uint16_t S,
                           uint16_t M,
                           uint16_t T,
                           CBFPtr progress_cb);

// Bias/Activation
uint16_t noodle_do_bias(float *output, float bias, uint16_t n);

// Pooling
uint16_t noodle_do_pooling(float *buffer, uint16_t W, uint16_t K,
                           uint16_t S, const char *fn);
uint16_t noodle_do_pooling(float *buffer, uint16_t W, uint16_t K,
                           uint16_t S, float *out_mem);
uint16_t noodle_do_pooling1d(float *buffer, uint16_t W, uint16_t K,
                             uint16_t S, const char *fn);

// Flatten
uint16_t noodle_flat(const char *in_fn, float *output, uint16_t V, uint16_t n_filters);
uint16_t noodle_flat(float *input, float *output, uint16_t V, uint16_t n_filters);

// Fully connected (overloads)
uint16_t noodle_fcn(int8_t *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const char *weight_fn, const char *bias_fn,
                    bool with_relu, CBFPtr progress_cb = NULL);
/** @overload noodle_fcn */
uint16_t noodle_fcn(float *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const char *weight_fn, const char *bias_fn,
                    bool with_relu = true, CBFPtr progress_cb = NULL);
/** @overload noodle_fcn */
uint16_t noodle_fcn(byte *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const char *weight_fn, const char *bias_fn,
                    bool with_relu = true, CBFPtr progress_cb = NULL);
/** @overload noodle_fcn */
uint16_t noodle_fcn(byte *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    float *output_buffer, const char *weight_fn, const char *bias_fn,
                    bool with_relu = true, CBFPtr progress_cb = NULL);
/** @overload noodle_fcn */
uint16_t noodle_fcn(float *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    float *output_buffer, const char *weight_fn, const char *bias_fn,
                    bool with_relu = true, CBFPtr progress_cb = NULL);
/** @overload noodle_fcn */
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    float *output_buffer, const char *weight_fn, const char *bias_fn,
                    bool with_relu = true, CBFPtr progress_cb = NULL);
/** @overload noodle_fcn */
uint16_t noodle_fcn(int8_t *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    float *output_buffer, const char *weight_fn, const char *bias_fn,
                    bool with_relu = true, CBFPtr progress_cb = NULL);
/** @overload noodle_fcn */
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const char *weight_fn, const char *bias_fn,
                    bool with_relu = true, CBFPtr progress_cb = NULL);

// Activations
uint16_t noodle_soft_max(float *input_output, uint16_t n);
uint16_t noodle_sigmoid(float *input_output, uint16_t n);

// 1D Convolution
uint16_t noodle_do_conv1d(float *input, float *kernel, uint16_t W, uint16_t K,
                          float *output_buffer, uint16_t P, uint16_t S);
uint16_t noodle_conv1d(float *input, float *output,
                       uint16_t n_inputs, uint16_t n_outputs,
                       const char *in_fn, const char *out_fn,
                       const char *weight_fn, const char *bias_fn,
                       uint16_t W, uint16_t P, uint16_t K, uint16_t S,
                       uint16_t M, uint16_t T, bool with_relu = true,
                       CBFPtr progress_cb = NULL);
/** @overload noodle_conv1d */
uint16_t noodle_conv1d(float *input, float *output,
                       uint16_t n_inputs, uint16_t n_outputs,
                       const char *in_fn, const char *out_fn,
                       const char *weight_fn, const char *bias_fn,
                       uint16_t W, uint16_t P, uint16_t K, uint16_t S,
                       bool with_relu = true, CBFPtr progress_cb = NULL);
