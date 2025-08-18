/**
 * @file noodle.h
 * @brief Lightweight CNN-style operations using SD card file storage for embedded systems.
 *
 * Supports convolution, pooling, flattening, and fully connected layers,
 * optimized for microcontrollers with low RAM by using file-based storage
 * for large intermediate activations.
 *
 * @authors
 * - Auralius Manurung — Universitas Telkom, Bandung
 * - Lisa Kristiana — ITENAS, Bandung
 *
 * @license MIT License
 */

#pragma once

#include "FS.h"
#include "SD_MMC.h"

/// Callback type for reporting operation progress.
typedef void (*CBFPtr)(float);

// ============================================================================
// SD CARD & FILE UTILITIES
// ============================================================================

/**
 * @brief Open a file for writing (overwrite if exists).
 * @param fn Filename to open.
 * @return File object opened in write mode.
 */
File noodle_open_file_for_write(const char* fn);

/**
 * @brief Read characters from file until a terminator or max length is reached.
 * @param file Reference to File object.
 * @param terminator Stop reading when this character is encountered.
 * @param buffer Buffer to store read data (null-terminated).
 * @param length Max number of bytes to store (including null terminator).
 * @return Number of bytes read (excluding null terminator).
 */
size_t noodle_read_bytes_until(File &file, char terminator, char *buffer, size_t length);

/**
 * @brief Initialize SD_MMC interface.
 * @param clk_pin Clock pin.
 * @param cmd_pin Command pin.
 * @param d0_pin Data0 pin.
 * @return true if initialization succeeds, false otherwise.
 */
bool noodle_sd_init(int clk_pin, int cmd_pin, int d0_pin);
bool noodle_sd_init();

/**
 * @brief Map a number to a two-letter lowercase string ("aa" to "zz").
 * @param number Number in range 1..26*26.
 * @param out Pointer to char[3] for result (2 letters + null terminator).
 */
void noodle_n2ll(uint16_t number, char *out);

void noodle_write_float(File &f, float d);
float noodle_read_float(File &f);
byte noodle_read_byte(File &f);
void noodle_write_byte(File &f, byte d);
void noodle_delete_file(const char *fn);

// ============================================================================
// MEMORY UTILITIES
// ============================================================================

float *noodle_create_buffer(uint16_t size);
void noodle_delete_buffer(float *buffer);
void noodle_reset_buffer(float *buffer, uint16_t n);

// ============================================================================
// GRID & ARRAY I/O
// ============================================================================

void noodle_array_to_file(float *array, const char *fn, uint16_t n);
void noodle_grid_to_file(byte *grid, const char *fn, uint16_t n);
float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P);

void noodle_array_from_file(const char *fn, float *buffer, uint16_t K);
void noodle_grid_from_file(const char *fn, byte *buffer, uint16_t K, bool transposed=false);
void noodle_grid_from_file(const char *fn, int8_t *buffer, uint16_t K, bool transposed=false);
void noodle_grid_from_file(const char *fn, float *buffer, uint16_t K, bool transposed=false);

// ============================================================================
// 2D CONVOLUTION
// ============================================================================

uint16_t noodle_do_conv(byte *grid, float *kernel, uint16_t K, uint16_t W,
                        float *output_buffer, uint16_t P, uint16_t S);

uint16_t noodle_conv(byte *grid, float *output_buffer,
                     uint16_t n_inputs, uint16_t n_outputs,
                     const char *in_fn, const char *out_fn,
                     const char *weight_fn, const char *bias_fn,
                     uint16_t W, uint16_t P, uint16_t K, uint16_t S,
                     uint16_t M, uint16_t T,
                     CBFPtr progress_cb=NULL);

// ============================================================================
// BIAS & ACTIVATION
// ============================================================================

uint16_t noodle_do_bias(float *output, float bias, uint16_t n);

// ============================================================================
// POOLING
// ============================================================================

uint16_t noodle_do_pooling(float *buffer, uint16_t W, uint16_t K,
                           uint16_t S, const char *fn);
uint16_t noodle_do_pooling1d(float *buffer, uint16_t W, uint16_t K,
                             uint16_t S, const char *fn);

// ============================================================================
// FLATTENING
// ============================================================================

uint16_t noodle_flat(const char *in_fn, float *output_buffer,
                     uint16_t V, uint16_t n_filters);

// ============================================================================
// FULLY CONNECTED LAYERS
// ============================================================================

/**
 * @brief Compute a fully connected (dense) layer.
 *
 * Supports multiple data type combinations for input/output, including:
 * - Memory-to-memory (float, byte, int8)
 * - Memory-to-file
 * - File-to-memory
 * - File-to-file
 *
 * Common parameters:
 * @param n_inputs Number of input elements.
 * @param n_outputs Number of output elements.
 * @param weight_fn Weight filename (n_outputs x n_inputs, row-major).
 * @param bias_fn Bias filename (n_outputs values).
 * @param with_relu Whether to apply ReLU after bias.
 * @param progress_cb Optional progress callback (0..100%).
 * @return Number of output elements.
 *
 * @overload int8 input → file output
 */
uint16_t noodle_fcn(int8_t *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const char *weight_fn, const char *bias_fn,
                    bool with_relu, CBFPtr progress_cb=NULL);

/**
 * @overload float input → file output
 */
uint16_t noodle_fcn(float *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const char *weight_fn, const char *bias_fn,
                    bool with_relu=true, CBFPtr progress_cb=NULL);

/**
 * @overload byte input → file output
 */
uint16_t noodle_fcn(byte *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const char *weight_fn, const char *bias_fn,
                    bool with_relu=true, CBFPtr progress_cb=NULL);

/**
 * @overload byte input → float output
 */
uint16_t noodle_fcn(byte *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    float *output_buffer, const char *weight_fn, const char *bias_fn,
                    bool with_relu=true, CBFPtr progress_cb=NULL);

/**
 * @overload float input → float output
 */
uint16_t noodle_fcn(float *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    float *output_buffer, const char *weight_fn, const char *bias_fn,
                    bool with_relu=true, CBFPtr progress_cb=NULL);

/**
 * @overload file input → float output
 */
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    float *output_buffer, const char *weight_fn, const char *bias_fn,
                    bool with_relu=true, CBFPtr progress_cb=NULL);

/**
 * @overload int8 input → float output
 */
uint16_t noodle_fcn(int8_t *input_buffer, uint16_t n_inputs, uint16_t n_outputs,
                    float *output_buffer, const char *weight_fn, const char *bias_fn,
                    bool with_relu=true, CBFPtr progress_cb=NULL);

/**
 * @overload file input → file output
 */
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const char *weight_fn, const char *bias_fn,
                    bool with_relu=true, CBFPtr progress_cb=NULL);

// ============================================================================
// ACTIVATIONS
// ============================================================================

uint16_t noodle_soft_max(float *input_output, uint16_t n);
uint16_t noodle_sigmoid(float *input_output, uint16_t n);

// ============================================================================
// 1D CONVOLUTION
// ============================================================================

/**
 * @brief Perform 1D convolution over a signal.
 * @param input Input signal (float array).
 * @param kernel Convolution kernel (float array).
 * @param W Input length.
 * @param K Kernel length.
 * @param output_buffer Output buffer (float array).
 * @param P Padding size.
 * @param S Stride.
 * @return Output length after convolution.
 */
uint16_t noodle_do_conv1d(float *input, float *kernel, uint16_t W, uint16_t K,
                          float *output_buffer, uint16_t P, uint16_t S);

/**
 * @brief 1D convolution pipeline with bias, activation, and pooling.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param in_fn Input filename pattern.
 * @param out_fn Output filename pattern.
 * @param weight_fn Convolution weight filename pattern.
 * @param bias_fn Bias filename.
 * @param W Input length.
 * @param P Padding size.
 * @param K Kernel size.
 * @param S Stride.
 * @param M Pooling kernel size.
 * @param T Pooling stride.
 * @param with_relu Whether to apply ReLU after bias.
 * @param progress_cb Optional progress callback.
 * @return Output length after pooling.
 *
 * @overload With pooling
 */
uint16_t noodle_conv1d(float *input, float *output_buffer,
                       uint16_t n_inputs, uint16_t n_outputs,
                       const char *in_fn, const char *out_fn,
                       const char *weight_fn, const char *bias_fn,
                       uint16_t W, uint16_t P, uint16_t K, uint16_t S,
                       uint16_t M, uint16_t T, bool with_relu=true,
                       CBFPtr progress_cb=NULL);

/**
 * @overload Without pooling
 */
uint16_t noodle_conv1d(float *input, float *output_buffer,
                       uint16_t n_inputs, uint16_t n_outputs,
                       const char *in_fn, const char *out_fn,
                       const char *weight_fn, const char *bias_fn,
                       uint16_t W, uint16_t P, uint16_t K, uint16_t S,
                       bool with_relu=true, CBFPtr progress_cb=NULL);
