/**
 * @file noodle.h
 * @brief Header file for lightweight CNN-style operations using SD card file storage on embedded systems.
 *
 * This module defines functions for performing convolution, pooling, flattening,
 * and fully connected neural network operations. It is optimized for use in
 * resource-constrained environments like Arduino or embedded systems.
 *
 * Features:
 * - File-based I/O using SdFat
 * - Support for float and byte data
 * - Functions for padding, convolution, bias/ReLU, max pooling, and FCN
 * - Optional progress callbacks for tracking computation
 *
 * @authors
 * - Auralius Manurung -- Universitas Telkom, Bandung (auralius.manurung@ieee.org)
 * - Lisa Kristiana -- ITENAS, Bandung (lisa@itenas.ac.id)
 *
 * @license MIT License
 */

#pragma once

#include <SdFat.h>

/// SD card object
extern SdFat SD;

/// Callback function pointer type for progress reporting
typedef void (*CBFPtr)(float);

/**
 * @brief Reads a float value from a file.
 * @param f Reference to the file.
 * @return Parsed float value.
 */
float noodle_read_float(File &f);

/**
 * @brief Deletes a file from the SD card.
 * @param fn Filename to delete.
 */
void noodle_delete_file(char *fn);

/**
 * @brief Allocates a buffer for float data.
 * @param size Number of float elements.
 * @return Pointer to the allocated buffer.
 */
float *noodle_create_buffer(uint16_t size);

/**
 * @brief Frees a previously allocated buffer.
 * @param buffer Pointer to the buffer to free.
 */
void noodle_delete_buffer(float *buffer);

/**
 * @brief Writes a grid of byte data to a file.
 * @param grid Byte array.
 * @param fn Filename to write to.
 * @param n Width/height of the square grid.
 */
void noodle_grid_to_file(byte *grid, char *fn, uint16_t n);

/**
 * @brief Gets a padded grid value.
 * @param grid Input byte grid.
 * @param i Row index.
 * @param j Column index.
 * @param W Width of the grid.
 * @param P Padding size.
 * @return Padded float value.
 */
float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P);

/**
 * @brief Applies bias and ReLU activation to the output.
 * @param output Pointer to output buffer.
 * @param bias Bias value to apply.
 * @param n Width/height of the square output.
 * @return Output size.
 */
uint16_t noodle_do_bias(float *output, float bias, uint16_t n);

/**
 * @brief Applies max pooling and stores the result in a file.
 * @param output Pointer to input/output buffer.
 * @param W Width of the input.
 * @param K Pooling kernel size.
 * @param S Pooling stride.
 * @param fn Output filename.
 * @return Output width.
 */
uint16_t noodle_do_pooling(float *output, uint16_t W, uint16_t K, uint16_t S, char *fn);

/**
 * @brief Performs 2D convolution on a byte grid.
 * @param grid Input byte grid.
 * @param kernel Convolution kernel.
 * @param K Kernel size.
 * @param W Input width.
 * @param output_buffer Buffer to store output.
 * @param P Padding size.
 * @param S Stride length.
 * @return Output size.
 */
uint16_t noodle_do_convolution(byte *grid, float *kernel, uint16_t K, uint16_t W, float *output_buffer, uint16_t P, uint16_t S);

/**
 * @brief Reads a byte matrix from file.
 * @param fn Filename to read from.
 * @param buffer Buffer to load data into.
 * @param K Width/height of the square matrix.
 * @param transposed Whether to transpose during reading.
 */
void noodle_read_from_file(char *fn, byte *buffer, uint16_t K, bool transposed = false);

/**
 * @brief Reads a float matrix from file.
 * @param fn Filename to read from.
 * @param buffer Buffer to load data into.
 * @param K Width/height of the square matrix.
 * @param transposed Whether to transpose during reading.
 */
void noodle_read_from_file(char *fn, float *buffer, uint16_t K, bool transposed = false);

/**
 * @brief Resets a float buffer to zero.
 * @param buffer Buffer to reset.
 * @param n Number of elements.
 */
void noodle_reset_buffer(float *buffer, uint16_t n);

/**
 * @brief Performs full convolution + bias + ReLU + pooling pipeline.
 * @param grid Input byte grid.
 * @param output_buffer Output buffer.
 * @param n_inputs Number of input feature maps.
 * @param n_filters Number of filters.
 * @param in_fn Input filename pattern.
 * @param out_fn Output filename pattern.
 * @param weight_fn Weight filename pattern.
 * @param bias_fn Bias filename.
 * @param W Input width.
 * @param P Padding size.
 * @param K Kernel size.
 * @param S Stride length.
 * @param M Pooling kernel size.
 * @param T Pooling stride length.
 * @param progress_cb Optional progress callback.
 * @return Output width.
 */
uint16_t noodle_conv(byte *grid, float *output_buffer, uint16_t n_inputs, uint16_t n_filters, char *in_fn, char *out_fn, char *weight_fn, char *bias_fn, uint16_t W, uint16_t P, uint16_t K, uint16_t S, uint16_t M, uint16_t T, CBFPtr progress_cb = NULL);

/**
 * @brief Flattens pooled feature maps into a 1D vector.
 * @param output_buffer Output buffer.
 * @param in_fn Input filename pattern.
 * @param V Width of pooled maps.
 * @param n_filters Number of filters.
 * @return Total flattened size.
 */
uint16_t noodle_flat(float *output_buffer, char *in_fn, uint16_t V, uint16_t n_filters);

/**
 * @brief Fully connected layer computation from buffer to file.
 * @param output_buffer Input data buffer.
 * @param n_inputs Number of input neurons.
 * @param n_outputs Number of output neurons.
 * @param out_fn Output filename.
 * @param weight_fn Weight filename.
 * @param bias_fn Bias filename.
 * @param progress_cb Optional progress callback.
 * @return Number of outputs.
 */
uint16_t noodle_fcn(float *output_buffer, uint16_t n_inputs, uint16_t n_outputs, char *out_fn, char *weight_fn, char *bias_fn, CBFPtr progress_cb = NULL);

/**
 * @brief Fully connected layer computation for byte input.
 * @param output_buffer Byte input buffer.
 * @param n_inputs Number of inputs.
 * @param n_outputs Number of outputs.
 * @param out_fn Output filename.
 * @param weight_fn Weight filename.
 * @param bias_fn Bias filename.
 * @param progress_cb Optional progress callback.
 * @return Number of outputs.
 */
uint16_t noodle_fcn(byte *output_buffer, uint16_t n_inputs, uint16_t n_outputs, char *out_fn, char *weight_fn, char *bias_fn, CBFPtr progress_cb = NULL);

/**
 * @brief Fully connected layer computation from file input.
 * @param in_fn Input filename.
 * @param n_inputs Number of inputs.
 * @param n_outputs Number of outputs.
 * @param output_buffer Output buffer.
 * @param weight_fn Weight filename.
 * @param bias_fn Bias filename.
 * @param progress_cb Optional progress callback.
 * @return Number of outputs.
 */
uint16_t noodle_fcn(char *in_fn, uint16_t n_inputs, uint16_t n_outputs, float *output_buffer, char *weight_fn, char *bias_fn, CBFPtr progress_cb = NULL);
