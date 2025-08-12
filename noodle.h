/**
 * @SdFile noodle.h
 * @brief Header SdFile for lightweight CNN-style operations using SD card SdFile storage on embedded systems.
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

// --- SdFile Utilities ---

/**
 * @brief Read bytes from an SdFile until a terminator is found or buffer is full.
 * @param file       Open SdFile to read from.
 * @param terminator Character marking the end of reading.
 * @param buffer     Output buffer (null-terminated on success).
 * @param length     Maximum number of bytes to read (including null terminator).
 * @return Number of bytes read (excluding null terminator).
 */
size_t noodle_read_bytes_until(SdFile &file, char terminator, char *buffer, size_t length);

/**
 * @brief Initialize SD functionalities.
 * @param cs_pin SPI enable pin.
 */
bool noodle_sd_init(uint8_t csPin);

/**
 * @brief Maps a number (1 to 26*26) to two letters (aa to zz).
 * @param number The numbers.
 * @param out The two letters
 */
void noodle_n2ll(uint16_t number, char *out);

/**
 * @brief Write a float value to a file.
 * @param f SdFile object to write to.
 * @param d Float value to write.
 */
void noodle_write_float(SdFile &f, float d);

/**
 * @brief Read a float value from a file.
 * @param f SdFile object to read from.
 * @return The float value read from the file.
 */
float noodle_read_float(SdFile &f);

/**
 * @brief Read a byte value from a file.
 * @param f SdFile object to read from.
 * @return The byte value read from the file.
 */
byte noodle_read_byte(SdFile &f);

/**
 * @brief Write an integer value to a file.
 * @param f SdFile object to write to.
 * @param d Integer value to write.
 */
void noodle_write_byte(SdFile &f, byte d);


/**
 * @brief Delete a SdFile from the SD card.
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
 * @brief Save a 1D byte array to file.
 * @param array Pointer to the byte grid.
 * @param fn Output filename.
 * @param n Grid dimension (n).
 */
void noodle_array_to_file(float *array,
                          const char *fn,
                          uint16_t n);

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

// --- SdFile Reading ---

void noodle_array_from_file(const char *fn,
                            float *buffer,
                            uint16_t K);

/**
 * @brief Read a byte matrix from file.
 * @param fn Input filename.
 * @param buffer Output buffer.
 * @param K Matrix size (K x K).
 * @param transposed Whether to transpose during read.
 */
void noodle_grid_from_file(const char *fn,
                           byte *buffer,
                           uint16_t K,
                           bool transposed = false);

/**
 * @brief Read an int8 matrix from file.
 * @param fn Input filename.
 * @param buffer Output buffer.
 * @param K Matrix size (K x K).
 * @param transposed Whether to transpose during read.
 */
void noodle_grid_from_file(const char *fn,
                           int8_t *buffer,
                           uint16_t K,
                           bool transposed = false);


/**
 * @brief Read a float matrix from file.
 * @param fn Input filename.
 * @param buffer Output buffer.
 * @param K Matrix size (K x K).
 * @param transposed Whether to transpose during read.
 */
void noodle_grid_from_file(const char *fn,
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
 * @brief Full 2D convolution pipeline with multiple inputs/outputs (byte inputs).
 * @overload Handles byte inputs.
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
                             const char *fn);

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

uint16_t noodle_fcn(int8_t *input_buffer, 
                    uint16_t n_inputs, 
                    uint16_t n_outputs, 
                    const char *out_fn, 
                    const char *weight_fn, 
                    const char *bias_fn, 
                    bool with_relu, 
                    CBFPtr progress_cb=NULL);

/**
 * @brief Compute a fully connected layer from float buffer input to SdFile output.
 * @overload Input from memory (float); output to SdFile (float)
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
 * @brief Compute a fully connected layer from byte buffer input to SdFile output.
 * @overload Input from memory (byte); output to SdFile (float)
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
 * @overload Input from memory (byte); output to memory (float)
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
 * @overload Input from memory (float); output to memory (float)
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
 * @brief Compute a fully connected layer from SdFile input to float buffer output.
 * @overload Input from SdFile (float); output to memory (float)
 */
uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output_buffer,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu = true,
                    CBFPtr progress_cb = NULL);

uint16_t noodle_fcn(int8_t *input_buffer,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output_buffer,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu = true,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Compute a fully connected layer from SdFile input to float buffer output.
 * @overload Input from SdFile (float); output to SdFile (float)
 */
uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu = true,
                    CBFPtr progress_cb = NULL);
// --- Activations ---

/**
 * @brief Apply softmax to vector.
 * @param input_output Input and output buffer.
 * @param n Number of elements.
 * @return n
 */
uint16_t noodle_soft_max(float *input_output, uint16_t n);

/**
 * @brief Apply sigmoid to vector.
 * @param input_output Input and output buffer.
 * @param n Number of elements.
 * @return n
 */
uint16_t noodle_sigmoid(float *input_output, uint16_t n);

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
uint16_t noodle_do_conv1d(float *input,
                          float *kernel,
                          uint16_t W,
                          uint16_t K,
                          float *output_buffer,
                          uint16_t P,
                          uint16_t S);

/**
 * @brief Full 1D convolution pipeline with bias, activation, and max pooling.
 *
 * This version loads input from file, applies convolution, adds bias,
 * applies optional ReLU activation, then performs 1D max pooling.
 *
 * @param input          Pointer to input signal array.
 * @param output_buffer  Output buffer.
 * @param n_inputs       Number of input channels.
 * @param n_outputs      Number of output channels.
 * @param in_fn          Input filename pattern (e.g., "in_%d.dat").
 * @param out_fn         Output filename pattern (e.g., "out_%d.dat").
 * @param weight_fn      Convolution weight filename pattern.
 * @param bias_fn        Bias filename.
 * @param W              Input signal length.
 * @param P              Padding size.
 * @param K              Kernel size.
 * @param S              Stride.
 * @param M              Pooling kernel size.
 * @param T              Pooling stride.
 * @param with_relu      Whether to apply ReLU after bias.
 * @param progress_cb    Optional progress callback.
 * @return Output signal length after pooling.
 */
uint16_t noodle_conv1d(float *input,
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
                       bool with_relu = true,
                       CBFPtr progress_cb = NULL);

/**
 * @brief 1D convolution pipeline without pooling.
 *
 * This version loads input from file, applies convolution, adds bias,
 * and applies optional ReLU activation — no pooling is performed.
 *
 * @param input          Pointer to input signal array.
 * @param output_buffer  Output buffer.
 * @param n_inputs       Number of input channels.
 * @param n_outputs      Number of output channels.
 * @param in_fn          Input filename pattern (e.g., "in_%d.dat").
 * @param out_fn         Output filename pattern (e.g., "out_%d.dat").
 * @param weight_fn      Convolution weight filename pattern.
 * @param bias_fn        Bias filename.
 * @param W              Input signal length.
 * @param P              Padding size.
 * @param K              Kernel size.
 * @param S              Stride.
 * @param with_relu      Whether to apply ReLU after bias.
 * @param progress_cb    Optional progress callback.
 * @return Output signal length after convolution.
 */
uint16_t noodle_conv1d(float *input,
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
                       bool with_relu = true,
                       CBFPtr progress_cb = NULL);
