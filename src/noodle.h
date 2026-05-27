/**
 * @file noodle.h
 * @brief Tiny CNN/ML primitives for microcontrollers with optional file streaming.
 * @ingroup noodle_public
 *
 * @details
 * Noodle provides small convolution, depthwise convolution, transpose convolution,
 * pooling, flattening, fully-connected, activation, batch-normalization, and tensor
 * utility routines. Most APIs are available in memory-backed forms, file-backed
 * forms, or mixed file/memory forms so models can run on boards with very limited
 * RAM.
 *
 * @section noodle_backends Filesystem Backends
 * Select exactly one backend macro before including this header:
 * `NOODLE_USE_SDFAT`, `NOODLE_USE_SD_MMC`, `NOODLE_USE_FFAT`,
 * `NOODLE_USE_LITTLEFS`, or `NOODLE_USE_NONE`. If no backend is selected,
 * `noodle_config.h` currently defaults to `NOODLE_USE_SDFAT`. See
 * `noodle_fs.h` for backend-specific types, path normalization, and open/remove
 * helpers.
 *
 * @section noodle_temp_buffers Temporary Buffers
 * Several convolution overloads reuse caller-owned temporary buffers. The exact
 * requirement is documented on each overload that uses them. Call
 * noodle_setup_temp_buffers() before using those APIs.
 *
 * Buffer roles:
 * - Buffer 1 (`b1`): file-input scratch. It holds one input plane/sequence.
 * - Buffer 2 (`b2`): accumulation scratch. It holds one pre-pooling output
 *   plane/sequence.
 *
 * Typical sizes:
 * - 2D input scratch: `W * W` input values (`byte` for byte-input convolution,
 *   `float` otherwise).
 * - 2D accumulator: `Vconv * Vconv` floats, where
 *   `Vconv = noodle_compute_V(K, W, P, S)`.
 * - 1D input scratch: `W` floats.
 * - 1D accumulator: `W` floats in the current 1D convolution overloads.
 *
 * @section noodle_layout Tensor Layout And Text Files
 * Unless a function says otherwise, tensors are channel-first and packed
 * contiguously:
 * - 2D feature maps: `[C][W][W]`, with each plane row-major.
 * - 1D feature maps: `[C][W]`.
 * - Convolution weights: `[O][I][K][K]` for 2D, `[O][I][K]` for 1D.
 * - Fully-connected weights: `[O][I]`.
 *
 * File-backed tensors and parameters are stored as ASCII numeric values, one
 * scalar per line, in the same packed order. Bias files contain one scalar per
 * output channel/neuron.
 *
 * @section noodle_pooling Pooling Mode
 * 2D pooling behavior is selected at compile time with `NOODLE_POOL_MODE` in
 * `noodle_config.h` (`NOODLE_POOL_NONE`, `NOODLE_POOL_MAX`, or
 * `NOODLE_POOL_MEAN`). The 1D pooling helper always performs max pooling.
 */

/**
 * @defgroup noodle_public Public API
 * Public functions, types, and configuration intended for application use.
 */

/**
 * @defgroup noodle_internal Internal Helpers
 * Helper routines used by Noodle implementations. These are documented for
 * maintainers, but application code normally should not call them directly.
 */
 
#pragma once

#include <stdint.h>
#ifdef ARDUINO
#include <Arduino.h>
#endif

#include "noodle_config.h"   // pooling mode & backend selection macros
#include "noodle_fs.h"       // NDL_File + NOODLE_FS wrappers

#ifndef ARDUINO
typedef unsigned char byte;    ///< Arduino-compatible byte alias for non-Arduino builds.
#endif
// ============================================================================

/**
 * @brief Activation applied after bias where supported.
 * @ingroup noodle_public
 *
 * Convolution and fully-connected layers support @ref ACT_NONE and @ref ACT_RELU.
 * Fully-connected memory outputs also support @ref ACT_SOFTMAX for final
 * classification heads.
 */
enum Activation : uint8_t {
  ACT_NONE    = 0,  ///< Do not apply an activation.
  ACT_RELU    = 1,  ///< Clamp negative values to zero.
  ACT_SOFTMAX = 2   ///< Normalize a vector with softmax where supported.
};

/**
 * @brief File-backed convolution parameters.
 * @ingroup noodle_public
 *
 * Weight files are read sequentially in output-major order. For normal 2D
 * convolution the layout is `[O][I][K][K]`; for 1D convolution it is `[O][I][K]`.
 * Bias files contain one scalar per output channel. The same type is also used
 * by depthwise convolution, where weights and biases are ordered by channel.
 */
struct Conv {
  uint16_t K  = 3;    ///< Kernel size (`K x K` for 2D, `K` taps for 1D).
  uint16_t P  = 0;    ///< Zero padding per side; `65535` requests SAME-style 2D padding.
  uint16_t S  = 1;    ///< Convolution stride.
  uint16_t OP = 0;    ///< Output padding for transpose convolution.

  const char *weight_fn = nullptr; ///< Weight filename.
  const char *bias_fn   = nullptr; ///< Bias filename.

  Activation act = ACT_RELU; ///< Activation applied after adding bias.
};

/**
 * @brief Memory-backed convolution parameters.
 * @ingroup noodle_public
 *
 * For normal 2D convolution, `weight` points to contiguous `[O][I][K][K]`
 * floats. For 1D convolution it points to `[O][I][K]`. For depthwise
 * convolution, weights are `[C][K][K]`. `bias` may be `nullptr` in overloads
 * that explicitly allow missing biases; in that case zero bias is used.
 */
struct ConvMem {
  uint16_t K  = 3;    ///< Kernel size (`K x K` for 2D, `K` taps for 1D).
  uint16_t P  = 0;    ///< Zero padding per side.
  uint16_t S  = 1;    ///< Convolution stride.
  uint16_t OP = 0;    ///< Output padding for transpose convolution.

  const float *weight = nullptr; ///< Pointer to packed weight values.
  const float *bias   = nullptr; ///< Pointer to packed bias values.

  Activation act = ACT_RELU; ///< Activation applied after adding bias.
};

/**
 * @brief Valid-pooling parameters.
 * @ingroup noodle_public
 *
 * Use `M = 1` and `T = 1` for identity pooling. 2D pooling uses the compile-time
 * `NOODLE_POOL_MODE`; 1D pooling uses max pooling.
 */
struct Pool {
  uint16_t M = 1;    ///< Pooling window size (`M x M` for 2D, `M` samples for 1D).
  uint16_t T = 1;    ///< Pooling stride.
};

/**
 * @brief Progress callback used by long-running routines.
 * @ingroup noodle_public
 * @param progress Normalized progress in the range `[0, 1]`.
 */
typedef void (*CBFPtr)(float progress);

/**
 * @brief Configure both reusable temporary buffers for convolution operations.
 * @ingroup noodle_public
 *
 * Call this before APIs that require scratch space. The buffers are stored as
 * raw pointers and are not owned by Noodle.
 *
 * @param b1 Input scratch buffer, used by overloads that stream file inputs.
 * @param b2 Accumulation/pre-pooling output scratch buffer.
 */
void noodle_setup_temp_buffers(void *b1, void *b2);

/**
 * @brief Configure only the reusable accumulation/output temporary buffer.
 * @ingroup noodle_public
 *
 * Use this overload for operations that do not need an input scratch buffer but
 * do need a plane-sized accumulator. This includes some memory-to-memory and
 * memory-to-file convolution overloads.
 *
 * @param b2 Accumulation/pre-pooling output scratch buffer.
 */
void noodle_setup_temp_buffers(void *b2);

/**
 * @name Filesystem Utilities
 */

/**
 * @brief Initialize the selected filesystem backend with SD_MMC 1-bit pins.
 * @ingroup noodle_public
 *
 * For `NOODLE_USE_SD_MMC`, this calls `SD_MMC.setPins(clk_pin, cmd_pin, d0_pin)`
 * and starts the bus in 1-bit mode. Other real backends ignore the pins and use
 * their default initializer.
 *
 * @param clk_pin SD_MMC clock pin.
 * @param cmd_pin SD_MMC command pin.
 * @param d0_pin SD_MMC D0 pin.
 * @return `true` when the selected backend initializes successfully.
 */
bool noodle_fs_init(uint8_t clk_pin, uint8_t cmd_pin, uint8_t d0_pin);

/**
 * @brief Initialize the selected filesystem backend with SD_MMC 4-bit pins.
 * @ingroup noodle_public
 *
 * For `NOODLE_USE_SD_MMC`, this configures CLK, CMD, and D0..D3 and starts the
 * bus in 4-bit mode. Other real backends ignore the pins and use their default
 * initializer.
 *
 * @param clk_pin SD_MMC clock pin.
 * @param cmd_pin SD_MMC command pin.
 * @param d0_pin SD_MMC D0 pin.
 * @param d1_pin SD_MMC D1 pin.
 * @param d2_pin SD_MMC D2 pin.
 * @param d3_pin SD_MMC D3 pin.
 * @return `true` when the selected backend initializes successfully.
 */
bool noodle_fs_init(uint8_t clk_pin, uint8_t cmd_pin, uint8_t d0_pin, uint8_t d1_pin, uint8_t d2_pin, uint8_t d3_pin);

/**
 * @brief Initialize the selected filesystem backend with default settings.
 * @ingroup noodle_public
 * @return `true` when the selected backend initializes successfully.
 */
bool noodle_fs_init();

/**
 * @brief Initialize the selected filesystem backend with a chip-select pin.
 * @ingroup noodle_public
 *
 * The `cs_pin` value is used by `NOODLE_USE_SDFAT`. Other backends ignore it and
 * use their default initializer.
 *
 * @param cs_pin SPI chip-select pin for SdFat.
 * @return `true` when the selected backend initializes successfully.
 */
bool noodle_fs_init(uint8_t cs_pin);

/**
 * @brief Read the first line from a text file.
 * @ingroup noodle_public
 *
 * The output is always NUL-terminated when @p maxlen is greater than zero. If the
 * file cannot be opened, @p line is set to an empty string.
 *
 * @param fn File name to read.
 * @param line Destination character buffer.
 * @param maxlen Destination capacity, including the NUL terminator.
 */
void noodle_read_top_line(const char* fn, char *line, size_t maxlen);

/**
 * @brief Delete a file using the selected filesystem backend.
 * @ingroup noodle_internal
 * @param fn File name to remove.
 */
void noodle_delete_file(const char *fn);

/**
 * @brief Read bytes from a file until a terminator or the buffer is full.
 * @ingroup noodle_public
 *
 * The terminator is consumed but not stored. @p buffer is always NUL-terminated
 * if @p length is greater than zero.
 *
 * @param file Open file handle.
 * @param terminator Character that ends the read.
 * @param buffer Destination character buffer.
 * @param length Destination capacity, including the NUL terminator.
 * @return Number of characters written, excluding the NUL terminator.
 */
size_t noodle_read_bytes_until(NDL_File &file, char terminator, char *buffer, size_t length);

/**
 * @name Scalar I/O Helpers
 */
 
/**
 * @brief Write a float using the configured file scalar format.
 * @ingroup noodle_internal
 * @param f Open output file.
 * @param d Value to write.
 */
void noodle_write_float(NDL_File &f, float d);

/**
 * @brief Read a float using the configured file scalar format.
 * @ingroup noodle_internal
 * @param f Open input file.
 * @return Parsed float value.
 */
float noodle_read_float(NDL_File &f);

/**
 * @brief Read a byte value using the configured file scalar format.
 * @ingroup noodle_internal
 * @param f Open input file.
 * @return Parsed byte value.
 */
byte noodle_read_byte(NDL_File &f);

/**
 * @brief Write a byte value using the configured file scalar format.
 * @ingroup noodle_internal
 * @param f Open output file.
 * @param d Value to write.
 */
void noodle_write_byte(NDL_File &f, byte d);

/**
 * @name Memory Utilities
 */

/**
 * @brief Allocate a raw buffer and return it as a float pointer.
 * @ingroup noodle_public
 *
 * This is a small compatibility wrapper around `malloc()`.
 *
 * @param size Number of bytes to allocate.
 * @return Pointer to the allocated memory, or `nullptr` on allocation failure.
 */
float *noodle_create_buffer(uint16_t size);

/**
 * @brief Free a buffer allocated by noodle_create_buffer().
 * @ingroup noodle_public
 * @param buffer Buffer pointer to free. Passing `nullptr` is allowed.
 */
void noodle_delete_buffer(float *buffer);

/**
 * @brief Fill a float buffer with zeros.
 * @ingroup noodle_internal
 * @param buffer Buffer to clear.
 * @param n Number of float elements to write.
 */
void noodle_reset_buffer(float *buffer, uint16_t n);

/**
 * @brief Return a channel plane from a packed `[Z][W][W]` tensor.
 * @ingroup noodle_internal
 * @param flat Base pointer to the contiguous tensor.
 * @param W Width and height of each 2D plane.
 * @param z Plane/channel index.
 * @return Pointer to the first element of plane @p z. No bounds checks are done.
 */
float* noodle_slice(float* flat, size_t W, size_t z);

/**
 * @brief Write a float array to a text file, one value per line.
 * @ingroup noodle_public
 * @param array Source array with @p n values.
 * @param fn Output filename. The file is opened and closed by this function.
 * @param n Number of values to write.
 */
void noodle_array_to_file(float *array, const char *fn, uint16_t n);

/**
 * @brief Write a float array to an already-open text file.
 * @ingroup noodle_internal
 * @param array Source array with @p n values.
 * @param fo Open output file handle. The handle remains open.
 * @param n Number of values to write.
 */
void noodle_array_to_file(float *array, NDL_File &fo, uint16_t n);

/**
 * @brief Write an `n x n` byte grid to a text file in row-major order.
 * @ingroup noodle_public
 * @param grid Source grid with `n * n` byte values.
 * @param fn Output filename. The file is opened and closed by this function.
 * @param n Grid width and height.
 */
void noodle_grid_to_file(byte *grid, const char *fn, uint16_t n);

/**
 * @brief Write an `n x n` byte grid to an already-open text file.
 * @ingroup noodle_internal
 * @param grid Source grid with `n * n` byte values.
 * @param fo Open output file handle. The handle remains open.
 * @param n Grid width and height.
 */
void noodle_grid_to_file(byte *grid, NDL_File &fo, uint16_t n);

/**
 * @brief Write an `n x n` float grid to a text file in row-major order.
 * @ingroup noodle_public
 * @param grid Source grid with `n * n` float values.
 * @param fn Output filename. The file is opened and closed by this function.
 * @param n Grid width and height.
 */
void noodle_grid_to_file(float *grid, const char *fn, uint16_t n);

/**
 * @brief Write an `n x n` float grid to an already-open text file.
 * @ingroup noodle_internal
 * @param grid Source grid with `n * n` float values.
 * @param fo Open output file handle. The handle remains open.
 * @param n Grid width and height.
 */
void noodle_grid_to_file(float *grid, NDL_File &fo, uint16_t n);

/**
 * @brief Read a float array from a text file.
 * @ingroup noodle_public
 * @param fn Input filename. The file is opened and closed by this function.
 * @param buffer Destination array with room for @p K floats.
 * @param K Number of values to read.
 */
void noodle_array_from_file(const char *fn, float *buffer, uint16_t K);

/**
 * @brief Read a float array from an already-open text file.
 * @ingroup noodle_public
 * @param fi Open input file handle. The handle remains open.
 * @param buffer Destination array with room for @p K floats.
 * @param K Number of values to read.
 */
void noodle_array_from_file(NDL_File &fi, float *buffer, uint16_t K);

/**
 * @brief Read a `K x K` grid into a byte buffer.
 * @ingroup noodle_public
 * @param fn Input filename. The file is opened and closed by this function.
 * @param buffer Destination buffer with room for `K * K` byte values.
 * @param K Grid width and height.
 */
void noodle_grid_from_file(const char *fn, byte *buffer, uint16_t K);

/**
 * @brief Read a `K x K` grid into a byte buffer from an already-open file.
 * @ingroup noodle_public
 * @param fi Open input file handle. The handle remains open.
 * @param buffer Destination buffer with room for `K * K` byte values.
 * @param K Grid width and height.
 */
void noodle_grid_from_file(NDL_File &fi, byte *buffer, uint16_t K);

/**
 * @brief Read a `K x K` grid into an int8 buffer.
 * @ingroup noodle_public
 * @param fn Input filename. The file is opened and closed by this function.
 * @param buffer Destination buffer with room for `K * K` int8 values.
 * @param K Grid width and height.
 */
void noodle_grid_from_file(const char *fn, int8_t *buffer, uint16_t K);

/**
 * @brief Read a `K x K` grid into an int8 buffer from an already-open file.
 * @ingroup noodle_public
 * @param fi Open input file handle. The handle remains open.
 * @param buffer Destination buffer with room for `K * K` int8 values.
 * @param K Grid width and height.
 */
void noodle_grid_from_file(NDL_File &fi, int8_t *buffer, uint16_t K);

/**
 * @brief Read a `K x K` grid into a float buffer.
 * @ingroup noodle_public
 * @param fn Input filename. The file is opened and closed by this function.
 * @param buffer Destination buffer with room for `K * K` float values.
 * @param K Grid width and height.
 */
void noodle_grid_from_file(const char *fn, float *buffer, uint16_t K);

/**
 * @brief Read a `K x K` grid into a float buffer from an already-open file.
 * @ingroup noodle_internal
 * @param fi Open input file handle. The handle remains open.
 * @param buffer Destination buffer with room for `K * K` float values.
 * @param K Grid width and height.
 */
void noodle_grid_from_file(NDL_File &fi, float *buffer, uint16_t K);

/**
 * @name Internal Pooling Helpers
 */

/**
 * @brief Apply valid 2D pooling and write the result to a file.
 * @ingroup noodle_internal
 *
 * When `NOODLE_POOL_MODE` is `NOODLE_POOL_NONE`, this writes the input map
 * unchanged and returns @p W. Otherwise it computes valid max or mean pooling.
 *
 * @param input Input map with `W * W` floats.
 * @param W Input width and height.
 * @param K Pooling window size.
 * @param S Pooling stride.
 * @param fn Output filename. The file is opened and closed by this function.
 * @return Output width, or `0` for invalid pooling parameters.
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);

/**
 * @brief Apply valid 2D pooling and write the result to an open file.
 * @ingroup noodle_internal
 * @param input Input map with `W * W` floats.
 * @param W Input width and height.
 * @param K Pooling window size.
 * @param S Pooling stride.
 * @param fo Open output file handle. The handle remains open.
 * @return Output width, or `0` for invalid pooling parameters.
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, NDL_File &fo);

/**
 * @brief Apply valid 2D pooling and write the result to memory.
 * @ingroup noodle_internal
 * @param input Input map with `W * W` floats.
 * @param W Input width and height.
 * @param K Pooling window size.
 * @param S Pooling stride.
 * @param output Destination buffer with room for the pooled map.
 * @return Output width, or @p W for identity/no pooling.
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, float *output);

/**
 * @brief Apply valid 1D max pooling and write the result to a file.
 * @ingroup noodle_internal
 * @param input Input vector with @p W floats.
 * @param W Input length.
 * @param K Pooling window size.
 * @param S Pooling stride.
 * @param fn Output filename. The file is opened and closed by this function.
 * @return Output length, `(@p W - @p K) / @p S + 1`.
 */
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);

/**
 * @brief Apply valid 1D max pooling and write the result to an open file.
 * @ingroup noodle_internal
 * @param input Input vector with @p W floats.
 * @param W Input length.
 * @param K Pooling window size.
 * @param S Pooling stride.
 * @param fo Open output file handle. The handle remains open.
 * @return Output length, `(@p W - @p K) / @p S + 1`.
 */
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S, NDL_File &fo);

/**
 * @name 2D Convolution
 *
 * Packed-file conventions:
 * - Input file: packed `[I][W][W]` planes.
 * - Output file: packed `[O][Vout][Vout]` planes.
 *
 * Padding is symmetric when `Conv::P` or `ConvMem::P` is an explicit value.
 * For 2D convolution, `P == 65535` requests SAME-style output sizing via
 * noodle_compute_V().
 *
 * Temp-buffer requirements are listed per overload. Some memory-to-memory
 * overloads still require buffer 2 because pooling needs a separate
 * pre-pooling accumulation plane.
 */

/**
 * @brief Run file-to-file 2D convolution on byte input feature maps.
 * @ingroup noodle_public
 *
 * Reads packed byte-valued input planes from @p in_fn, streams file-backed
 * weights/biases from @p conv, applies activation and pooling, and writes packed
 * float output planes to @p out_fn.
 *
 * @warning Requires both temporary buffers. Call
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b1` must hold at least
 * `W * W` byte values. `b2` must hold at least `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param in_fn Input file containing packed `[I][W][W]` planes.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param out_fn Output file for packed `[O][Vout][Vout]` planes.
 * @param W Input width and height.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */
uint16_t noodle_conv_byte(const char *in_fn,
                          uint16_t n_inputs,
                          uint16_t n_outputs,
                          const char *out_fn,
                          uint16_t W,
                          const Conv &conv,
                          const Pool &pool,
                          CBFPtr progress_cb = NULL);

/**
 * @brief Run file-to-file 2D convolution on float input feature maps.
 * @ingroup noodle_public
 *
 * Reads packed float input planes from @p in_fn, streams file-backed
 * weights/biases from @p conv, applies activation and pooling, and writes packed
 * output planes to @p out_fn.
 *
 * @warning Requires both temporary buffers. Call
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b1` must hold at least
 * `W * W` floats. `b2` must hold at least `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param in_fn Input file containing packed `[I][W][W]` planes.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param out_fn Output file for packed `[O][Vout][Vout]` planes.
 * @param W Input width and height.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run file-to-memory 2D convolution on float input feature maps.
 * @ingroup noodle_public
 *
 * The output tensor is channel-first and packed as `[O][Vout][Vout]`.
 *
 * @warning Requires both temporary buffers. Call
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b1` must hold at least
 * `W * W` floats. `b2` must hold at least `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param in_fn Input file containing packed `[I][W][W]` planes.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param output Destination buffer for packed output planes.
 * @param W Input width and height.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run memory-to-file 2D convolution with file-backed parameters.
 * @ingroup noodle_public
 *
 * @warning Requires the second temporary buffer. Call
 * `noodle_setup_temp_buffers(b2)` or `noodle_setup_temp_buffers(b1, b2)` before
 * calling. `b2` must hold at least `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param input Input tensor in packed `[I][W][W]` layout.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param out_fn Output file for packed `[O][Vout][Vout]` planes.
 * @param W Input width and height.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run memory-to-file 2D convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * @warning Requires the second temporary buffer. Call
 * `noodle_setup_temp_buffers(b2)` or `noodle_setup_temp_buffers(b1, b2)` before
 * calling. `b2` must hold at least `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param input Input tensor in packed `[I][W][W]` layout.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param out_fn Output file for packed `[O][Vout][Vout]` planes.
 * @param W Input width and height.
 * @param conv Memory-backed convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const ConvMem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run memory-to-memory 2D convolution with file-backed parameters.
 * @ingroup noodle_public
 *
 * @warning Requires the second temporary buffer, even though both input and
 * output tensors are in memory. Call
 * `noodle_setup_temp_buffers(b2)` or `noodle_setup_temp_buffers(b1, b2)` before
 * calling. `b2` must hold at least `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param input Input tensor in packed `[I][W][W]` layout.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param output Destination tensor in packed `[O][Vout][Vout]` layout.
 * @param W Input width and height.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run memory-to-memory 2D convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * @warning Requires the second temporary buffer, even though the tensors and
 * parameters are in memory. Call `noodle_setup_temp_buffers(b2)` or
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b2` must hold at least
 * `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param input Input tensor in packed `[I][W][W]` layout.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param output Destination tensor in packed `[O][Vout][Vout]` layout.
 * @param W Input width and height.
 * @param conv Memory-backed convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const ConvMem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);
               
/**
 * @name 1D Convolution
 *
 * 1D convolution uses packed `[C][W]` tensors. `Conv::K`/`ConvMem::K` is the
 * kernel length and @p W is the input sequence length. Temp-buffer
 * requirements are listed per overload.
 */
 
/**
 * @brief Run file-to-file 1D convolution with file-backed parameters and pooling.
 * @ingroup noodle_public
 *
 * Reads packed `[I][W]` input samples from @p in_fn, accumulates one output
 * sequence per output channel, applies bias/activation, applies valid 1D max
 * pooling, and appends packed `[O][Vout]` output samples to @p out_fn.
 *
 * @warning Requires both temporary buffers. Call
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b1` must hold at least
 * `W` floats for one input sequence, and `b2` must hold at least `W` floats for
 * one accumulated output sequence.
 *
 * @param in_fn Input file containing packed `[I][W]` sequences.
 * @param n_inputs Number of input channels.
 * @param out_fn Output file for packed `[O][Vout]` sequences.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output length after pooling.
 */
uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const Conv &conv,
                       const Pool &pool,
                       CBFPtr progress_cb=NULL);

/**
 * @brief Run file-to-file 1D convolution with file-backed parameters.
 * @ingroup noodle_public
 *
 * This overload does not pool. It writes packed raw convolution outputs after
 * bias/activation.
 *
 * @warning Requires both temporary buffers. Call
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b1` must hold at least
 * `W` floats for one input sequence, and `b2` must hold at least `W` floats for
 * one accumulated output sequence.
 *
 * @param in_fn Input file containing packed `[I][W]` sequences.
 * @param n_inputs Number of input channels.
 * @param out_fn Output file for packed `[O][V]` sequences.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv File-backed convolution parameters.
 * @param progress_cb Optional normalized progress callback.
 * @return Output length before pooling, `(@p W - K + 2P) / S + 1`.
 */
uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const Conv &conv,
                       CBFPtr progress_cb=NULL);

/**
 * @brief Run file-to-file 1D convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * This overload does not pool.
 *
 * @warning Requires both temporary buffers. Call
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b1` must hold at least
 * `W` floats for one input sequence, and `b2` must hold at least `W` floats for
 * one accumulated output sequence.
 *
 * @param in_fn Input file containing packed `[I][W]` sequences.
 * @param n_inputs Number of input channels.
 * @param out_fn Output file for packed `[O][V]` sequences.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv Memory-backed convolution parameters.
 * @param progress_cb Optional normalized progress callback.
 * @return Output length before pooling, or `0` if temporary buffers are missing.
 */
uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb);
                       
/**
 * @brief Run memory-to-memory 1D convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * This overload does not use temporary buffers and does not pool.
 *
 * @param in Input tensor in packed `[I][W]` layout.
 * @param n_inputs Number of input channels.
 * @param out Destination tensor in packed `[O][V]` layout.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv Memory-backed convolution parameters.
 * @param progress_cb Optional normalized progress callback.
 * @return Output length before pooling.
 */
uint16_t noodle_conv1d(float *in,
                       uint16_t n_inputs,
                       float *out,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb=NULL);

/**
 * @brief Run memory-to-file 1D convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * This overload does not pool.
 *
 * @warning Requires the second temporary buffer as an output accumulator. Call
 * `noodle_setup_temp_buffers(b2)` or `noodle_setup_temp_buffers(b1, b2)` before
 * calling. `b2` must hold at least `W` floats.
 *
 * @param in Input tensor in packed `[I][W]` layout.
 * @param n_inputs Number of input channels.
 * @param out_fn Output file for packed `[O][V]` sequences.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv Memory-backed convolution parameters.
 * @param progress_cb Optional normalized progress callback.
 * @return Output length before pooling, or `0` if the accumulator buffer is missing.
 */
uint16_t noodle_conv1d(float *in,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb);

/**
 * @brief Run file-to-memory 1D convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * This overload does not pool. The output buffer is packed by output channel,
 * with each channel occupying `Vmax = (W - K + 2P) / S + 1` values.
 *
 * @warning Requires the first temporary buffer. Call
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b1` must hold at least
 * `W` floats for one input sequence. This overload does not use `b2`.
 *
 * @param in_fn Input file containing packed `[I][W]` sequences.
 * @param n_inputs Number of input channels.
 * @param out Destination tensor in packed `[O][Vmax]` layout.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv Memory-backed convolution parameters.
 * @param progress_cb Optional normalized progress callback.
 * @return Output length before pooling.
 */
uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       float *out,              // packed output: [O][Vmax]
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb);

/**
 * @name Internal Convolution Helpers
 */
                       
/**
 * @brief Accumulate one padded/strided 1D convolution into an output buffer.
 * @ingroup noodle_internal
 *
 * The output buffer is not cleared; results are added to existing values.
 *
 * @param input Input sequence with @p W floats.
 * @param kernel Kernel with @p K floats.
 * @param W Input sequence length.
 * @param K Kernel length.
 * @param output Accumulator buffer with room for the computed output sequence.
 * @param P Zero padding per side.
 * @param S Stride.
 * @return Output length, `(@p W - @p K + 2 * @p P) / @p S + 1`.
 */
uint16_t noodle_do_conv1d(float *input, float *kernel, uint16_t W, uint16_t K,
                          float *output, uint16_t P, uint16_t S);

/**
 * @brief Accumulate one padded/strided 2D convolution from a byte grid.
 * @ingroup noodle_internal
 *
 * The output buffer is not cleared; results are added to existing values.
 *
 * @param grid Input grid with `W * W` byte values.
 * @param kernel Kernel with `K * K` floats.
 * @param K Kernel width and height.
 * @param W Input width and height.
 * @param output Accumulator buffer with room for the computed output plane.
 * @param P Zero padding per side.
 * @param S Stride.
 * @return Output width and height.
 */
uint16_t noodle_do_conv(byte *grid, const float *kernel, uint16_t K, uint16_t W,
                        float *output, uint16_t P, uint16_t S);

/**
 * @brief Accumulate one padded/strided 2D convolution from a float grid.
 * @ingroup noodle_internal
 * @param grid Input grid with `W * W` float values.
 * @param kernel Kernel with `K * K` floats.
 * @param K Kernel width and height.
 * @param W Input width and height.
 * @param output Accumulator buffer with room for the computed output plane.
 * @param P Zero padding per side.
 * @param S Stride.
 * @return Output width and height.
 */
uint16_t noodle_do_conv(float *grid, const float *kernel, uint16_t K, uint16_t W,
                        float *output, uint16_t P, uint16_t S);

/**
 * @brief Add bias to a square map in place and apply ReLU.
 * @ingroup noodle_internal
 *
 * This legacy helper always applies ReLU after adding @p bias.
 *
 * @param output Accumulator buffer with `n * n` floats.
 * @param bias Scalar bias to add to every element.
 * @param n Map width and height.
 * @return @p n.
 */
uint16_t noodle_do_bias(float *output, float bias, uint16_t n);

/**
 * @brief Add bias to a square map in place and apply an activation.
 * @ingroup noodle_internal
 * @param output Accumulator buffer with `n * n` floats.
 * @param bias Scalar bias to add to every element.
 * @param n Map width and height.
 * @param act Activation to apply; currently @ref ACT_NONE and @ref ACT_RELU are handled.
 * @return @p n.
 */
uint16_t noodle_do_bias_act(float *output, float bias, uint16_t n, Activation act);

/**
 * @brief Read a byte grid sample with zero padding.
 * @ingroup noodle_internal
 * @param grid Input grid with `W * W` byte values.
 * @param i Row coordinate in padded space.
 * @param j Column coordinate in padded space.
 * @param W Input width and height.
 * @param P0 Top/left padding.
 * @param P1 Bottom/right padding.
 * @return Grid value converted to float, or `0.0f` outside the unpadded grid.
 */
static inline float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P0, int16_t P1) __attribute__((always_inline));

/**
 * @brief Read a float grid sample with zero padding.
 * @ingroup noodle_internal
 * @param grid Input grid with `W * W` float values.
 * @param i Row coordinate in padded space.
 * @param j Column coordinate in padded space.
 * @param W Input width and height.
 * @param P0 Top/left padding.
 * @param P1 Bottom/right padding.
 * @return Grid value, or `0.0f` outside the unpadded grid.
 */
static inline float noodle_get_padded_x(float *grid, int16_t i, int16_t j, int16_t W, int16_t P0, int16_t P1) __attribute__((always_inline));


/**
 * @name Activations
 */
 
/**
 * @brief Apply numerically stabilized softmax in place.
 * @ingroup noodle_public
 * @param input_output Vector of @p n logits; replaced with probabilities.
 * @param n Number of elements.
 * @return @p n.
 */
uint16_t noodle_soft_max(float *input_output, uint16_t n);

/**
 * @brief Apply logistic sigmoid to each element in place.
 * @ingroup noodle_public
 * @param input_output Vector of @p n values to transform.
 * @param n Number of elements.
 * @return @p n.
 */
uint16_t noodle_sigmoid(float *input_output, uint16_t n);

/**
 * @brief Compute the logistic sigmoid of one float.
 * @ingroup noodle_public
 * @param x Input value.
 * @return `1 / (1 + exp(-x))`, computed with a stable branch.
 */
float noodle_sigmoidf(float x);

/**
 * @brief Apply logistic sigmoid in place using the legacy logit helper name.
 * @ingroup noodle_public
 * @param input_output Vector of @p n values to transform.
 * @param n Number of elements.
 * @return @p n.
 */
uint16_t noodle_logit(float *input_output, uint16_t n);

/**
 * @brief Apply ReLU to each element in place.
 * @ingroup noodle_public
 * @param input_output Vector of @p n values to transform.
 * @param n Number of elements.
 * @return @p n.
 */
uint16_t noodle_relu(float *input_output, uint16_t n);

/**
 * @brief Legacy file-backed fully-connected parameter bundle.
 * @ingroup noodle_public
 *
 * New code should generally use @ref FCNFile, which has the same fields.
 */
struct FCN {
  const char *weight_fn = nullptr; ///< Weight filename with row-major `[O][I]` values.
  const char *bias_fn   = nullptr; ///< Bias filename with one scalar per output.
  Activation act = ACT_RELU;       ///< Activation applied after each output is computed.
};

/**
 * @brief File-backed fully-connected parameter bundle.
 * @ingroup noodle_public
 *
 * The weight file is read in row-major `[O][I]` order and the bias file contains
 * one scalar per output neuron.
 */
struct FCNFile {
  const char *weight_fn = nullptr; ///< Weight filename with row-major `[O][I]` values.
  const char *bias_fn   = nullptr; ///< Bias filename with one scalar per output.
  Activation act = ACT_RELU;       ///< Activation applied after each output is computed.
};

/**
 * @brief Memory-backed fully-connected parameter bundle.
 * @ingroup noodle_public
 *
 * `weight` points to row-major `[O][I]` floats. `bias` points to
 * @p n_outputs floats and may be `nullptr` for zero bias.
 */
struct FCNMem {
  const float *weight = nullptr; ///< Pointer to row-major `[O][I]` weights.
  const float *bias   = nullptr; ///< Pointer to output biases, or `nullptr` for zero bias.
  Activation act = ACT_RELU;     ///< Activation applied after each output is computed.
};

/**
 * @name Fully Connected Layers
 */

/**
 * @brief Run a memory-to-file fully-connected layer with int8 inputs.
 * @ingroup noodle_public
 *
 * Computes `y = W*x + b`, applies ReLU when requested, and writes one output
 * float per line to @p out_fn.
 *
 * @param input Input vector with @p n_inputs int8 values.
 * @param n_inputs Number of input values.
 * @param n_outputs Number of output neurons.
 * @param out_fn Output filename.
 * @param fcn File-backed weights, biases, and activation mode.
 * @param progress_cb Optional normalized progress callback.
 * @return @p n_outputs.
 */
uint16_t noodle_fcn(const int8_t *input, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);


/**
 * @brief Run a file-to-file fully-connected layer with file-backed parameters.
 * @ingroup noodle_public
 *
 * For each output neuron, the input file is rewound, one dot product is
 * accumulated, ReLU is applied when requested, and one float is written.
 *
 * @param in_fn Input filename containing @p n_inputs floats.
 * @param n_inputs Number of input values.
 * @param n_outputs Number of output neurons.
 * @param out_fn Output filename.
 * @param fcn File-backed weights, biases, and activation mode.
 * @param progress_cb Optional normalized progress callback.
 * @return @p n_outputs.
 */
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);


/**
 * @brief Run a memory-to-memory fully-connected layer with memory-backed parameters.
 * @ingroup noodle_public
 *
 * Applies ReLU or softmax when requested by @p fcn.
 *
 * @param input Input vector with @p n_inputs floats.
 * @param n_inputs Number of input values.
 * @param n_outputs Number of output neurons.
 * @param output Destination vector with room for @p n_outputs floats.
 * @param fcn Memory-backed weights, biases, and activation mode.
 * @param progress_cb Optional normalized progress callback.
 * @return @p n_outputs.
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNMem &fcn,
                    CBFPtr progress_cb = NULL);


/**
 * @brief Run a memory-to-memory fully-connected layer with byte inputs.
 * @ingroup noodle_public
 *
 * Weights and biases are read from files. Applies ReLU or softmax when requested.
 *
 * @param input Input vector with @p n_inputs byte values.
 * @param n_inputs Number of input values.
 * @param n_outputs Number of output neurons.
 * @param output Destination vector with room for @p n_outputs floats.
 * @param fcn File-backed weights, biases, and activation mode.
 * @param progress_cb Optional normalized progress callback.
 * @return @p n_outputs.
 */
uint16_t noodle_fcn(const byte *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);


/**
 * @brief Run a memory-to-memory fully-connected layer with int8 inputs.
 * @ingroup noodle_public
 *
 * Weights and biases are read from files. Applies ReLU or softmax when requested.
 *
 * @param input Input vector with @p n_inputs int8 values.
 * @param n_inputs Number of input values.
 * @param n_outputs Number of output neurons.
 * @param output Destination vector with room for @p n_outputs floats.
 * @param fcn File-backed weights, biases, and activation mode.
 * @param progress_cb Optional normalized progress callback.
 * @return @p n_outputs.
 */
uint16_t noodle_fcn(const int8_t *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);


/**
 * @brief Run a file-to-memory fully-connected layer with file-backed parameters.
 * @ingroup noodle_public
 *
 * Applies ReLU or softmax when requested by @p fcn.
 *
 * @param in_fn Input filename containing @p n_inputs floats.
 * @param n_inputs Number of input values.
 * @param n_outputs Number of output neurons.
 * @param output Destination vector with room for @p n_outputs floats.
 * @param fcn File-backed weights, biases, and activation mode.
 * @param progress_cb Optional normalized progress callback.
 * @return @p n_outputs.
 */
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Run a memory-to-memory fully-connected layer with float inputs.
 * @ingroup noodle_public
 *
 * Weights and biases are read from files. Applies ReLU or softmax when requested.
 *
 * @param input Input vector with @p n_inputs floats.
 * @param n_inputs Number of input values.
 * @param n_outputs Number of output neurons.
 * @param output Destination vector with room for @p n_outputs floats.
 * @param fcn File-backed weights, biases, and activation mode.
 * @param progress_cb Optional normalized progress callback.
 * @return @p n_outputs.
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb);

/**
 * @brief Run a memory-to-file fully-connected layer with float inputs.
 * @ingroup noodle_public
 *
 * Computes `y = W*x + b`, applies ReLU when requested, and writes one output
 * float per line to @p out_fn.
 *
 * @param input Input vector with @p n_inputs floats.
 * @param n_inputs Number of input values.
 * @param n_outputs Number of output neurons.
 * @param out_fn Output filename.
 * @param fcn File-backed weights, biases, and activation mode.
 * @param progress_cb Optional normalized progress callback.
 * @return @p n_outputs.
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const FCNFile &fcn,
                    CBFPtr progress_cb);

/**
 * @name Tensor Reshaping And Reductions
 */

/**
 * @brief Flatten a packed file tensor into HWC-like memory order.
 * @ingroup noodle_public
 *
 * Reads an input file in packed `[C][V][V]` order and writes @p output in
 * interleaved spatial-major order: `output[pixel * n_filters + channel]`.
 *
 * @param in_fn Input filename containing packed `[C][V][V]` values.
 * @param output Destination vector with room for `V * V * n_filters` floats.
 * @param V Input width and height.
 * @param n_filters Number of channels.
 * @return Number of output values written.
 */
uint16_t noodle_flat(const char *in_fn, float *output, uint16_t V, uint16_t n_filters);

/**
 * @brief Flatten a packed memory tensor into HWC-like memory order.
 * @ingroup noodle_public
 *
 * Reads @p input in packed `[C][V][V]` order and writes @p output as
 * `output[pixel * n_filters + channel]`.
 *
 * @param input Input tensor in packed `[C][V][V]` layout.
 * @param output Destination vector with room for `V * V * n_filters` floats.
 * @param V Input width and height.
 * @param n_filters Number of channels.
 * @return Number of output values written.
 */
uint16_t noodle_flat(float *input, float *output, uint16_t V, uint16_t n_filters);

/**
 * @brief Apply global average pooling in place.
 * @ingroup noodle_public
 *
 * Reads a channel-first `[C][W][W]` tensor from @p inout and writes the per-channel
 * means into the first @p C positions of the same buffer.
 *
 * @param inout Input/output buffer containing a packed `[C][W][W]` tensor.
 * @param C Number of channels.
 * @param W Width and height of each channel plane.
 * @return @p C.
 */
uint16_t noodle_gap(float *inout,
                    uint16_t C,
                    uint16_t W);

/**
 * @brief Find the maximum value and index in a float vector.
 * @ingroup noodle_public
 * @param input Input vector with @p n floats.
 * @param n Number of values to scan.
 * @param max_val Reference that receives the maximum value.
 * @param max_idx Reference that receives the index of the maximum value.
 */
void noodle_find_max(float *input,
                     uint16_t n,
                     float &max_val,
                     uint16_t &max_idx);

/** 
 *  @name 2D Depth-wise Convolution 
 *
 * Temp-buffer requirements are listed per overload. Buffer 2 is used as the
 * pre-pooling accumulation plane for depthwise convolution.
 */

/**
 * @brief Run file-to-file 2D depthwise convolution with file-backed parameters.
 * @ingroup noodle_public
 *
 * The input and output are packed channel-first files. Each channel is convolved
 * with its own `K x K` kernel, then bias, activation, and pooling are applied.
 *
 * @warning Requires both temporary buffers. Call
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b1` must hold at least
 * `W * W` floats. `b2` must hold at least `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param in_fn Input file containing packed `[C][W][W]` planes.
 * @param n_channels Number of input/output channels.
 * @param out_fn Output file for packed `[C][Vout][Vout]` planes.
 * @param W Input width and height.
 * @param conv File-backed depthwise convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */
uint16_t noodle_dwconv_float(const char *in_fn,
                             uint16_t n_channels,
                             const char *out_fn,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb);


/**
 * @brief Run memory-to-memory 2D depthwise convolution with file-backed parameters.
 * @ingroup noodle_public
 *
 * @warning Requires the second temporary buffer. Call
 * `noodle_setup_temp_buffers(b2)` or `noodle_setup_temp_buffers(b1, b2)` before
 * calling. `b2` must hold at least `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param input Input tensor in packed `[C][W][W]` layout.
 * @param n_channels Number of input/output channels.
 * @param output Destination tensor in packed `[C][Vout][Vout]` layout.
 * @param W Input width and height.
 * @param conv File-backed depthwise convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */
uint16_t noodle_dwconv_float(float *input,
                             uint16_t n_channels,
                             float *output,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb);
/**
 * @brief Run memory-to-memory 2D depthwise convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * @warning Requires the second temporary buffer, even though the tensors and
 * parameters are in memory. Call `noodle_setup_temp_buffers(b2)` or
 * `noodle_setup_temp_buffers(b1, b2)` before calling. `b2` must hold at least
 * `Vconv * Vconv` floats, where
 * `Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S)`.
 *
 * @param input Input tensor in packed `[C][W][W]` layout.
 * @param n_channels Number of input/output channels.
 * @param output Destination tensor in packed `[C][Vout][Vout]` layout.
 * @param W Input width and height.
 * @param conv Memory-backed depthwise convolution parameters.
 * @param pool Pooling parameters applied after bias/activation.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width after pooling.
 */                             
uint16_t noodle_dwconv_float(float *input,
                             uint16_t n_channels,
                             float *output,
                             uint16_t W,
                             const ConvMem &conv,
                             const Pool &pool,
                             CBFPtr progress_cb);                            

/**
 * @brief Split packed batch-normalization parameters into per-field pointers.
 * @ingroup noodle_internal
 *
 * @p bn_params must be packed as `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param bn_params Packed batch-normalization parameter array.
 * @param C Number of channels.
 * @param gamma Receives pointer to the gamma/scale values.
 * @param beta Receives pointer to the beta/offset values.
 * @param mean Receives pointer to the mean values.
 * @param var Receives pointer to the variance values.
 */
void noodle_unpack_bn_params(const float *bn_params,
                             uint16_t C,
                             const float **gamma,
                             const float **beta,
                             const float **mean,
                             const float **var);

/**
 * @name Batch Normalization
 */

/**
 * @brief Apply batch normalization to a packed channel-first tensor in place.
 * @ingroup noodle_public
 *
 * Computes `gamma[c] * (x - mean[c]) / sqrt(var[c] + eps) + beta[c]` for each
 * element of channel @p c.
 *
 * @param x Tensor in packed `[C][W][W]` layout; modified in place.
 * @param C Number of channels.
 * @param W Width and height of each channel plane.
 * @param gamma Per-channel scale values.
 * @param beta Per-channel offset values.
 * @param mean Per-channel means.
 * @param var Per-channel variances.
 * @param eps Small constant added to variance.
 * @return @p W.
 */
uint16_t noodle_bn(float *x,
                   uint16_t C,
                   uint16_t W,
                   const float *gamma,
                   const float *beta,
                   const float *mean,
                   const float *var,
                   float eps=1e-3);

/**
 * @brief Apply batch normalization followed by ReLU in place.
 * @ingroup noodle_public
 * @param x Tensor in packed `[C][W][W]` layout; modified in place.
 * @param C Number of channels.
 * @param W Width and height of each channel plane.
 * @param gamma Per-channel scale values.
 * @param beta Per-channel offset values.
 * @param mean Per-channel means.
 * @param var Per-channel variances.
 * @param eps Small constant added to variance.
 * @return @p W.
 */
uint16_t noodle_bn_relu(float *x,
                        uint16_t C,
                        uint16_t W,
                        const float *gamma,
                        const float *beta,
                        const float *mean,
                        const float *var,
                        float eps=1e-3);

/**
 * @brief Apply batch normalization using packed parameter storage.
 * @ingroup noodle_public
 *
 * @p bn_params must be packed as `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor in packed `[C][W][W]` layout; modified in place.
 * @param C Number of channels.
 * @param W Width and height of each channel plane.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small constant added to variance.
 * @return @p W.
 */ 
uint16_t noodle_bn(float *x,
                   uint16_t C,
                   uint16_t W,
                   const float *bn_params,
                   float eps=1e-3);

/**
 * @brief Apply batch normalization and ReLU using packed parameter storage.
 * @ingroup noodle_public
 *
 * @p bn_params must be packed as `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor in packed `[C][W][W]` layout; modified in place.
 * @param C Number of channels.
 * @param W Width and height of each channel plane.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small constant added to variance.
 * @return @p W.
 */
uint16_t noodle_bn_relu(float *x,
                        uint16_t C,
                        uint16_t W,
                        const float *bn_params,
                        float eps=1e-3);

/**
 * @brief Compute the square output width for 2D convolution.
 * @ingroup noodle_public
 *
 * For explicit symmetric padding, returns `(W - K + 2 * P) / S + 1`. When
 * @p P is `65535`, treats the convolution as SAME-style padding and returns
 * `ceil(W / S)`.
 *
 * @param K Kernel width and height.
 * @param W Input width and height.
 * @param P Padding per side, or `65535` for SAME-style output sizing.
 * @param S Stride.
 * @return Computed output width.
 */
uint16_t noodle_compute_V(uint16_t K, 
                          uint16_t W , 
                          uint16_t P, 
                          uint16_t S);

/**
 * @brief Compute the square output width and resolved 2D convolution padding.
 * @ingroup noodle_public
 *
 * For explicit symmetric padding, sets @p P0 and @p P1 to @p P and returns
 * `(W - K + 2 * P) / S + 1`. When @p P is `65535`, treats the convolution as
 * SAME-style padding, returns `ceil(W / S)`, and splits the required total
 * padding between top/left (@p P0) and bottom/right (@p P1).
 *
 * @param K Kernel width and height.
 * @param W Input width and height.
 * @param P Padding per side, or `65535` for SAME-style output sizing.
 * @param S Stride.
 * @param P0 Reference that receives the top/left padding.
 * @param P1 Reference that receives the bottom/right padding.
 * @return Computed output width.
 */
uint16_t noodle_compute_V_and_P(uint16_t K, 
                                uint16_t W , 
                                uint16_t P, 
                                uint16_t S, 
                                uint16_t &P0, 
                                uint16_t &P1);


/**
 * @brief Apply valid max pooling to a packed tensor in place.
 * @ingroup noodle_public
 *
 * The input and output layout is packed `[C][W][W]`; after pooling, channels are
 * compacted in place as `[C][Wo][Wo]`.
 *
 * @param inplace Tensor buffer to pool in place.
 * @param W Input width and height.
 * @param C Number of channels.
 * @param pool Pooling parameters.
 * @return Output width, or `0` for invalid input parameters.
 */
uint16_t noodle_valid_max_pool(float *inplace,
                               uint16_t W,
                               uint16_t C,
                               const Pool &pool);


/**
 * @name 2D Transpose Convolution
 *
 * Memory-backed transpose convolution uses packed `[I][W][W]` input,
 * packed `[O][Vt][Vt]` output, and weights in `[O][I][K][K]` order.
 *
 * The output width formula is `(W - 1) * S + K - 2 * P + OP`.
 */

/**
 * @brief Compute the output width for 2D transpose convolution.
 * @ingroup noodle_public
 *
 * @param K Kernel width and height.
 * @param W Input width and height.
 * @param P Padding per side.
 * @param S Stride.
 * @param OP Output padding.
 * @return Output width, or `0` if the computed size is invalid.
 */
uint16_t noodle_compute_Vt(uint16_t K,
                           uint16_t W,
                           uint16_t P,
                           uint16_t S,
                           uint16_t OP = 0);
uint16_t noodle_compute_Vt_and_P(uint16_t K,
                                 uint16_t W,
                                 uint16_t P,
                                 uint16_t S,
                                 uint16_t OP,
                                 uint16_t &P0,
                                 uint16_t &P1);

/**
 * @brief Accumulate one 2D transpose-convolution input plane into an output plane.
 * @ingroup noodle_internal
 *
 * The output buffer is not cleared; results are added to existing values.
 *
 * @param input Input plane with `W * W` floats.
 * @param kernel Kernel with `K * K` floats.
 * @param K Kernel width and height.
 * @param W Input width and height.
 * @param output Accumulator plane with room for `Vt * Vt` floats.
 * @param P Padding per side.
 * @param S Stride.
 * @param OP Output padding.
 * @return Output width, or `0` if the computed size is invalid.
 */
uint16_t noodle_do_conv_transpose(float *input,
                                  const float *kernel,
                                  uint16_t K,
                                  uint16_t W,
                                  float *output,
                                  uint16_t P,
                                  uint16_t S,
                                  uint16_t OP = 0);

/**
 * @brief Run memory-to-memory 2D transpose convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * Bias and activation are applied after all input channels have been accumulated
 * for each output channel.
 *
 * @param input Input tensor in packed `[I][W][W]` layout.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param output Destination tensor in packed `[O][Vt][Vt]` layout.
 * @param W Input width and height.
 * @param conv Memory-backed transpose-convolution parameters.
 * @param progress_cb Optional normalized progress callback.
 * @return Output width, or `0` if the input or parameters are invalid.
 */
uint16_t noodle_conv_transpose_float(float *input,
                                     uint16_t n_inputs,
                                     uint16_t n_outputs,
                                     float *output,
                                     uint16_t W,
                                     const ConvMem &conv,
                                     CBFPtr progress_cb = NULL);
