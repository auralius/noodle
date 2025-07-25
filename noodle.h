/**
 * @file noodle.h
 * @brief Header file for lightweight CNN-style operations using SD card file storage on embedded systems.
 *
 * This module defines functions for performing convolution, pooling, flattening,
 * and fully connected neural network operations. 
 *
 * @authors
 * - Auralius Manurung -- Universitas Telkom, Bandung (auralius.manurung@ieee.org)
 * - Lisa Kristiana -- ITENAS, Bandung (lisa@itenas.ac.id)
 *
 * @license MIT License
 */

#pragma once

#include <SdFat.h>

/// Callback function pointer type for progress reporting
typedef void (*CBFPtr)(float);

// --- File Utilities ---
/**
 * @brief Initialze SD functionalities.
 * @param cs_pin SPI enable pin.
 */
bool noodle_sd_init(uint8_t csPin);

/**
 * @brief Write a float value to a file.
 * @param f File object to write to.
 * @param d Float value to write.
 */
void noodle_write_float(File &f, float d);

/**
 * @brief Read a float value from a file.
 * @param f File object to read from.
 * @return The float value read from the file.
 */
float noodle_read_float(File &f);

/**
 * @brief Delete a file from the SD card.
 * @param fn Filename to delete.
 */
void noodle_delete_file(const char *fn);

// --- Memory Management ---

/**
 * @brief Allocate a buffer of float values.
 * @param size Number of floats to allocate.
 * @return Pointer to the allocated buffer.
 */
float *noodle_create_buffer(uint16_t size);

/**
 * @brief Free a previously allocated float buffer.
 * @param buffer Pointer to the buffer to free.
 */
void noodle_delete_buffer(float *buffer);

/**
 * @brief Reset a float buffer to zeros.
 * @param buffer Pointer to the buffer.
 * @param n Number of elements to reset.
 */
void noodle_reset_buffer(float *buffer, uint16_t n);

// --- Input/Output Grid Handling ---

/**
 * @brief Save a 2D byte grid to file.
 * @param grid Pointer to the byte grid.
 * @param fn Output filename.
 * @param n Grid dimension (n x n).
 */
void noodle_grid_to_file(byte *grid, 
                         const char *fn, 
                         uint16_t n);

/**
 * @brief Get a value from a padded grid.
 * @param grid Pointer to the byte grid.
 * @param i Row index.
 * @param j Column index.
 * @param W Grid width.
 * @param P Padding size.
 * @return Padded float value.
 */
float noodle_get_padded_x(byte *grid, 
                          int16_t i, 
                          int16_t j, 
                          int16_t W, 
                          int16_t P);

// --- File Reading ---

/**
 * @brief Read a byte matrix from file.
 * @param fn Input filename.
 * @param buffer Output buffer.
 * @param K Matrix size (K x K).
 * @param transposed Whether to transpose during read.
 */
void noodle_read_from_file(const char *fn, 
                           byte *buffer, 
                           uint16_t K, 
                           bool transposed = false);

/**
 * @brief Read a float matrix from file.
 * @param fn Input filename.
 * @param buffer Output buffer.
 * @param K Matrix size (K x K).
 * @param transposed Whether to transpose during read.
 */
void noodle_read_from_file(const char *fn, 
                           float *buffer, 
                           uint16_t K, 
                           bool transposed = false);

// --- Convolution Operations ---

/**
 * @brief Perform 2D convolution over input byte grid.
 * @param grid Input byte array.
 * @param kernel Convolution kernel.
 * @param K Kernel size.
 * @param W Input width.
 * @param output_buffer Output float buffer.
 * @param P Padding.
 * @param S Stride.
 * @return Output feature map size.
 */
uint16_t noodle_do_conv(byte *grid, 
                        float *kernel, 
                        uint16_t K, 
                        uint16_t W, 
                        float *output_buffer, 
                        uint16_t P, 
                        uint16_t S);

/**
 * @brief Full 2D convolution pipeline with multiple inputs/outputs.
 *
 * Combines convolution, bias addition, ReLU activation, and max pooling over a set of input/output channels.
 * Uses files for intermediate activations and weights.
 *
 * @param grid Temporary grid.
 * @param output_buffer Output float buffer.
 * @param n_inputs Number of input maps.
 * @param n_outputs Number of output maps.
 * @param in_fn Input filename pattern.
 * @param out_fn Output filename pattern.
 * @param weight_fn Weight filename pattern.
 * @param bias_fn Bias filename.
 * @param W Input width.
 * @param P Padding.
 * @param K Kernel size.
 * @param S Stride.
 * @param M Pooling kernel.
 * @param T Pooling stride.
 * @param progress_cb Optional callback.
 * @return Output feature map size.
 */
uint16_t noodle_conv(byte *grid, 
                     float *output_buffer, 
                     uint16_t n_inputs, 
                     uint16_t n_outputs, 
                     const char *in_fn, 
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

// --- Bias & Activation ---

/**
 * @brief Apply bias and ReLU to feature map.
 * @param output Pointer to output buffer.
 * @param bias Bias value.
 * @param n Size of output map.
 * @return Output map size.
 */
uint16_t noodle_do_bias(float *output, float bias, uint16_t n);

// --- Pooling Operations ---

/**
 * @brief Perform 2D max pooling.
 * @param buffer Input feature map.
 * @param W Width of input.
 * @param K Pooling kernel.
 * @param S Stride.
 * @param fn Output filename.
 * @return Output width.
 */
uint16_t noodle_do_pooling(float *buffer, 
                           uint16_t W, 
                           uint16_t K, 
                           uint16_t S, 
                           const char *fn);

/**
 * @brief Perform 1D max pooling.
 * @param buffer Input signal.
 * @param W Length of signal.
 * @param K Pooling kernel size.
 * @param S Stride.
 * @param fn Output filename.
 * @return Output length.
 */
uint16_t noodle_do_pooling1d(float *buffer, 
                             uint16_t W, 
                             uint16_t K, 
                             uint16_t S, 
                             char *fn);

// --- Flattening ---

/**
 * @brief Flatten multiple 2D maps into 1D.
 * @param in_fn Input filename pattern.
 * @param output_buffer Output buffer.
 * @param V Width of pooled maps.
 * @param n_filters Number of filters.
 * @return Total length of flat vector.
 */
uint16_t noodle_flat(const char *in_fn, 
                     float *output_buffer, 
                     uint16_t V, 
                     uint16_t n_filters);

// --- Fully Connected Layers ---

/**
 * @brief Compute a fully connected layer from float buffer input to file output.
 */
uint16_t noodle_fcn(float *input_buffer, 
                    uint16_t n_inputs, 
                    uint16_t n_outputs, 
                    const char *out_fn, 
                    const char *weight_fn, 
                    const char *bias_fn, 
                    bool with_relu = true, 
                    CBFPtr progress_cb = NULL);

/**
 * @brief Compute a fully connected layer from byte buffer input to file output.
 */
uint16_t noodle_fcn(byte *input_buffer, 
                    uint16_t n_inputs, 
                    uint16_t n_outputs, 
                    const char *out_fn, 
                    const char *weight_fn, 
                    const char *bias_fn, 
                    bool with_relu = true, 
                    CBFPtr progress_cb = NULL);

/**
 * @brief Compute a fully connected layer from byte buffer input to float buffer output.
 */
uint16_t noodle_fcn(byte *input_buffer, 
                    uint16_t n_inputs, 
                    uint16_t n_outputs, 
                    float *output_buffer, 
                    const char *weight_fn, 
                    const char *bias_fn, 
                    bool with_relu = true, 
                    CBFPtr progress_cb = NULL);

/**
 * @brief Compute a fully connected layer from float buffer input to float buffer output.
 */
uint16_t noodle_fcn(float *input_buffer, 
                    uint16_t n_inputs, 
                    uint16_t n_outputs, 
                    float *output_buffer, 
                    const char *weight_fn, 
                    const char *bias_fn, 
                    bool with_relu = true, 
                    CBFPtr progress_cb = NULL);

/**
 * @brief Compute a fully connected layer from file input to float buffer output.
 */
uint16_t noodle_fcn(const char *in_fn, 
                    uint16_t n_inputs, 
                    uint16_t n_outputs, 
                    float *output_buffer, 
                    const char *weight_fn, 
                    const char *bias_fn, 
                    bool with_relu = true, 
                    CBFPtr progress_cb = NULL);

// --- Softmax ---

/**
 * @brief Apply softmax to vector.
 * @param input_output Input and output buffer.
 * @param n Number of elements.
 * @return n
 */
uint16_t noodle_soft_max(float *input_output, uint16_t n);

// --- 1D Convolution and Pooling ---

/**
 * @brief Perform 1D convolution over a signal.
 * @param input Input signal.
 * @param kernel Convolution kernel.
 * @param W Length of signal.
 * @param K Kernel size.
 * @param output_buffer Output buffer.
 * @param P Padding.
 * @param S Stride.
 * @return Output signal length.
 */
uint16_t noodle_do_conv1d(byte *input, 
                          float *kernel, 
                          uint16_t W, 
                          uint16_t K, 
                          float *output_buffer, 
                          uint16_t P, 
                          uint16_t S);

/**
 * @brief Full 1D convolution pipeline.
 *
 * This includes input loading from file, convolution, bias, ReLU, and max pooling.
 *
 * @param input Input signal buffer.
 * @param output_buffer Output buffer.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param in_fn Input filename pattern.
 * @param out_fn Output filename pattern.
 * @param weight_fn Weight filename pattern.
 * @param bias_fn Bias filename.
 * @param W Input length.
 * @param P Padding.
 * @param K Kernel size.
 * @param S Stride.
 * @param M Pooling kernel size.
 * @param T Pooling stride.
 * @param progress_cb Optional callback.
 * @return Output signal length.
 */
uint16_t noodle_conv1d(byte *input, 
                       float *output_buffer, 
                       uint16_t n_inputs, 
                       uint16_t n_outputs, 
                       const char *in_fn, 
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
