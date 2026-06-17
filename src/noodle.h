/**
 * @file noodle.h
 * @brief Public Noodle API for small CNN/ML inference on microcontrollers.
 * @ingroup noodle_public
 *
 * Noodle provides compact convolution, depthwise convolution, transpose
 * convolution, fully connected, pooling, activation, batch-normalization, and
 * tensor-layout helpers. The public RAM-to-RAM layer functions use
 * NoodleBuffer so tensor storage can grow automatically. Raw float pointers are
 * kept for small math utilities, simple file utilities, and private
 * implementation helpers declared in noodle_internal.h.
 *
 * Unless a function says otherwise, feature maps use channel-first packed
 * storage:
 * - 2D tensors: `[C][W][W]`.
 * - 1D tensors: `[C][W]`.
 * - 2D convolution weights: `[O][I][K][K]`.
 * - 1D convolution weights: `[O][I][K]`.
 * - Depthwise convolution weights: `[C][K][K]`.
 * - Fully connected weights: `[O][I]`.
 *
 * File-backed APIs use the scalar encoding selected by `NOODLE_FILE_FORMAT`.
 * Text mode stores one scalar at a time; binary mode stores raw scalar bytes in
 * the same packed order.
 *
 * Internal convolution scratch buffers are allocated and grown on demand by
 * default. The legacy noodle_setup_temp_buffers() overloads can still install
 * fixed caller-owned buffers; when they are used, Noodle treats those pointers
 * as external memory with unknown capacity and never resizes or frees them.
 */

/**
 * @defgroup noodle_public Public API
 * Public functions, types, and configuration intended for application code.
 */

/**
 * @defgroup noodle_api Implementation API
 * Source files and private helpers that implement the public Noodle API.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef ARDUINO
#include <Arduino.h>
#endif

#ifndef ARDUINO
typedef unsigned char byte;  ///< Arduino-compatible byte alias for non-Arduino builds.
#ifndef NULL
#define NULL 0
#endif
#endif

#include "noodle_config.h"
#include "noodle_fs.h"
#include "noodle_buffer.h"
#include "noodle_tensor.h"

#if defined(__AVR__)
#include <avr/pgmspace.h>
#endif

/**
 * @brief Read a float from normal memory or near AVR PROGMEM.
 * @ingroup noodle_public
 *
 * On AVR this uses pgm_read_float_near(). On other platforms it indexes normal
 * memory directly, which lets the same PROGMEM-backed wrappers compile across
 * targets.
 *
 * @param p Base pointer to packed float values.
 * @param idx Element index to read.
 * @return Float value at @p idx.
 */
static inline float noodle_pgm_float(const float *p, uint32_t idx) {
#if defined(__AVR__)
  return pgm_read_float_near(p + idx);
#else
  return p[idx];
#endif
}

// ============================================================
// Public types
// ============================================================

/**
 * @brief Activation applied after bias where supported.
 * @ingroup noodle_public
 */
enum Activation : uint8_t {
  ACT_NONE    = 0,  ///< Do not apply an activation.
  ACT_RELU    = 1,  ///< Clamp negative values to zero.
  ACT_SOFTMAX = 2   ///< Normalize a final output vector where supported.
};

/**
 * @brief File-backed convolution parameter bundle.
 * @ingroup noodle_public
 *
 * Weight and bias files are read sequentially. For 2D convolution, weights use
 * `[O][I][K][K]`; for 1D convolution, `[O][I][K]`; for depthwise convolution,
 * `[C][K][K]`. Bias files contain one scalar per output channel, or one scalar
 * per channel for depthwise convolution.
 *
 * For transpose convolution with explicit padding, callers choose OP to match
 * the desired output width: `V = (W - 1) * S - 2 * P + K + OP`.
 */
struct Conv {
  uint16_t K  = 3;       ///< Kernel width, or tap count for 1D convolution.
  uint16_t P  = 0;       ///< Padding per side; `65535` requests SAME-style 2D padding.
  uint16_t S  = 1;       ///< Convolution stride.
  uint16_t OP = 0;       ///< User-computed output padding for transpose convolution.

  const char *weight_fn = nullptr;  ///< Weight filename.
  const char *bias_fn   = nullptr;  ///< Bias filename.

  Activation act = ACT_RELU;        ///< Activation applied after adding bias.
  uint16_t O = 0;                   ///< Optional output channel count for tensor wrappers.
};

/**
 * @brief File-backed convolution parameter bundle alias.
 * @ingroup noodle_public
 *
 * This carries the same layout and semantics as Conv and is kept for code that
 * prefers an explicit file-backed name.
 *
 * For transpose convolution with explicit padding, callers choose OP to match
 * the desired output width: `V = (W - 1) * S - 2 * P + K + OP`.
 */
struct ConvFile {
  uint16_t K  = 3;       ///< Kernel width, or tap count for 1D convolution.
  uint16_t P  = 0;       ///< Padding per side; `65535` requests SAME-style 2D padding.
  uint16_t S  = 1;       ///< Convolution stride.
  uint16_t OP = 0;       ///< User-computed output padding for transpose convolution.

  const char *weight_fn = nullptr;  ///< Weight filename.
  const char *bias_fn   = nullptr;  ///< Bias filename.

  Activation act = ACT_RELU;        ///< Activation applied after adding bias.
  uint16_t O = 0;                   ///< Optional output channel count for tensor wrappers.
};

/**
 * @brief Memory-backed convolution parameter bundle.
 * @ingroup noodle_public
 *
 * `weight` points to packed floats in the same order as file-backed weights:
 * `[O][I][K][K]` for 2D convolution, `[O][I][K]` for 1D convolution,
 * `[C][K][K]` for depthwise convolution, and `[O][I][K][K]` for transpose
 * convolution. `bias` may be `nullptr` in overloads that allow zero bias.
 *
 * For transpose convolution with explicit padding, callers choose OP to match
 * the desired output width: `V = (W - 1) * S - 2 * P + K + OP`.
 */
struct ConvMem {
  uint16_t K  = 3;       ///< Kernel width, or tap count for 1D convolution.
  uint16_t P  = 0;       ///< Padding per side; `65535` requests SAME-style 2D padding.
  uint16_t S  = 1;       ///< Convolution stride.
  uint16_t OP = 0;       ///< User-computed output padding for transpose convolution.

  const float *weight = nullptr;    ///< Pointer to packed weight values.
  const float *bias   = nullptr;    ///< Pointer to packed bias values, or nullptr.

  Activation act = ACT_RELU;        ///< Activation applied after adding bias.
  uint16_t O = 0;                   ///< Optional output channel count for tensor wrappers.
};

/**
 * @brief Near-PROGMEM convolution parameter bundle.
 * @ingroup noodle_public
 *
 * This is intended for small or medium AVR flash arrays that can be addressed by
 * pgm_read_float_near(). The packed layouts match ConvMem.
 */
struct ConvProgmem {
  uint16_t K  = 3;       ///< Kernel width.
  uint16_t P  = 0;       ///< Padding per side; `65535` requests SAME-style 2D padding.
  uint16_t S  = 1;       ///< Convolution stride.
  uint16_t OP = 0;       ///< Reserved output padding field for layout parity.

  const float *weight = nullptr;    ///< PROGMEM pointer to packed weights.
  const float *bias   = nullptr;    ///< PROGMEM pointer to biases, or nullptr.

  Activation act = ACT_RELU;        ///< Activation applied after adding bias.
  uint16_t O = 0;                   ///< Optional output channel count for tensor wrappers.
};

/**
 * @brief Valid-pooling parameter bundle.
 * @ingroup noodle_public
 *
 * Use `M = 1` and `T = 1` for identity pooling. 2D pooling uses the compile-time
 * `NOODLE_POOL_MODE`. Memory-backed 1D pooling computes mean pooling only when
 * `NOODLE_POOL_MODE == NOODLE_POOL_MEAN`; otherwise it computes max pooling.
 * File-backed 1D pooling helpers perform max pooling.
 */
struct Pool {
  uint16_t M = 1;  ///< Pool window size, `M x M` for 2D or `M` samples for 1D.
  uint16_t T = 1;  ///< Pool stride.
};

/**
 * @brief Progress callback used by long-running layer routines.
 * @ingroup noodle_public
 * @param progress Normalized progress, usually in the range `[0, 1]`.
 */
typedef void (*CBFPtr)(float progress);

/**
 * @brief File-backed fully connected parameter bundle.
 * @ingroup noodle_public
 *
 * Weight files are read in row-major `[O][I]` order. Bias files contain one
 * scalar per output neuron.
 */
struct FCN {
  const char *weight_fn = nullptr;  ///< Weight filename with `[O][I]` values.
  const char *bias_fn   = nullptr;  ///< Bias filename with one scalar per output.
  Activation act = ACT_RELU;        ///< Activation applied after each output.
  uint16_t O = 0;                   ///< Optional output count for tensor wrappers.
};

/**
 * @brief File-backed fully connected parameter bundle alias.
 * @ingroup noodle_public
 */
struct FCNFile {
  const char *weight_fn = nullptr;  ///< Weight filename with `[O][I]` values.
  const char *bias_fn   = nullptr;  ///< Bias filename with one scalar per output.
  Activation act = ACT_RELU;        ///< Activation applied after each output.
  uint16_t O = 0;                   ///< Optional output count for tensor wrappers.
};

/**
 * @brief Memory-backed fully connected parameter bundle.
 * @ingroup noodle_public
 */
struct FCNMem {
  const float *weight = nullptr;    ///< Pointer to row-major `[O][I]` weights.
  const float *bias   = nullptr;    ///< Pointer to output biases, or nullptr.
  Activation act = ACT_RELU;        ///< Activation applied after each output.
  uint16_t O = 0;                   ///< Optional output count for tensor wrappers.
};

/**
 * @brief Far-PROGMEM fully connected parameter bundle for AVR.
 * @ingroup noodle_public
 *
 * `weight_far` and `bias_far` are far flash addresses such as values produced by
 * pgm_get_far_address(). On non-AVR targets, FCNProgmem overloads compile but
 * return 0.
 */
struct FCNProgmem {
  uint32_t weight_far = 0;  ///< Far flash address of row-major `[O][I]` weights.
  uint32_t bias_far   = 0;  ///< Far flash address of biases, or 0 for zero bias.
  uint8_t act         = ACT_RELU;  ///< Activation mode using Activation values.
  uint16_t O          = 0;         ///< Optional output count for tensor wrappers.
};

// ============================================================
// Filesystem and scalar I/O
// ============================================================

/**
 * @brief Initialize SD_MMC with explicit 1-bit pins, or default-init other backends.
 * @ingroup noodle_public
 * @param clk_pin SD_MMC clock pin.
 * @param cmd_pin SD_MMC command pin.
 * @param d0_pin SD_MMC D0 pin.
 * @return true when the selected backend initializes successfully.
 */
bool noodle_fs_init(uint8_t clk_pin, uint8_t cmd_pin, uint8_t d0_pin);

/**
 * @brief Initialize SD_MMC with explicit 4-bit pins, or default-init other backends.
 * @ingroup noodle_public
 * @param clk_pin SD_MMC clock pin.
 * @param cmd_pin SD_MMC command pin.
 * @param d0_pin SD_MMC D0 pin.
 * @param d1_pin SD_MMC D1 pin.
 * @param d2_pin SD_MMC D2 pin.
 * @param d3_pin SD_MMC D3 pin.
 * @return true when the selected backend initializes successfully.
 */
bool noodle_fs_init(uint8_t clk_pin, uint8_t cmd_pin, uint8_t d0_pin,
                    uint8_t d1_pin, uint8_t d2_pin, uint8_t d3_pin);

/**
 * @brief Initialize the selected filesystem backend with default settings.
 * @ingroup noodle_public
 * @return true when the selected backend initializes successfully.
 */
bool noodle_fs_init();

/**
 * @brief Initialize the selected filesystem backend with an SPI chip-select pin.
 * @ingroup noodle_public
 *
 * The chip-select pin is used by the SdFat backend. Other real backends ignore
 * it and use their default initializer.
 *
 * @param cs_pin SPI chip-select pin for SdFat.
 * @return true when the selected backend initializes successfully.
 */
bool noodle_fs_init(uint8_t cs_pin);

#if defined(NOODLE_USE_SDFAT)
/**
 * @brief Initialize SdFat with an explicit SPI bus and clock speed.
 * @ingroup noodle_public
 *
 * Use this overload when the SD card is connected to a non-default SPI bus or
 * custom SPI pins that have already been configured on @p spi.
 *
 * @param cs_pin SPI chip-select pin for the SD card.
 * @param spi SPI bus instance used by SdFat.
 * @param sck_mhz Requested SD SPI clock in MHz.
 * @return true when SdFat initializes successfully.
 */
bool noodle_fs_init(uint8_t cs_pin, SPIClass &spi, uint8_t sck_mhz);
#endif

/**
 * @brief Read the first line from a text file.
 * @ingroup noodle_public
 *
 * The destination is NUL-terminated when @p maxlen is greater than zero. If the
 * file cannot be opened, @p line is set to an empty string.
 *
 * @param fn File to read.
 * @param line Destination character buffer.
 * @param maxlen Destination capacity, including the trailing NUL.
 */
void noodle_read_top_line(const char *fn, char *line, size_t maxlen);

/**
 * @brief Delete a file through the selected filesystem backend.
 * @ingroup noodle_public
 * @param fn File to remove.
 */
void noodle_delete_file(const char *fn);

/**
 * @brief Read bytes until a terminator or until the buffer is full.
 * @ingroup noodle_public
 *
 * The terminator is consumed but not stored. @p buffer is NUL-terminated when
 * @p length is greater than zero.
 *
 * @param file Open file handle.
 * @param terminator Character that ends the read.
 * @param buffer Destination character buffer.
 * @param length Destination capacity, including the trailing NUL.
 * @return Number of characters written, excluding the trailing NUL.
 */
size_t noodle_read_bytes_until(NDL_File &file, char terminator,
                               char *buffer, size_t length);

/**
 * @brief Write a float using `NOODLE_FILE_FORMAT`.
 * @ingroup noodle_public
 * @param f Open output file.
 * @param d Value to write.
 */
void noodle_write_float(NDL_File &f, float d);

/**
 * @brief Read a float using `NOODLE_FILE_FORMAT`.
 * @ingroup noodle_public
 * @param f Open input file.
 * @return Parsed or decoded float value.
 */
float noodle_read_float(NDL_File &f);

/**
 * @brief Read a byte using `NOODLE_FILE_FORMAT`.
 * @ingroup noodle_public
 * @param f Open input file.
 * @return Parsed or decoded byte value.
 */
byte noodle_read_byte(NDL_File &f);

/**
 * @brief Write a byte using `NOODLE_FILE_FORMAT`.
 * @ingroup noodle_public
 * @param f Open output file.
 * @param d Value to write.
 */
void noodle_write_byte(NDL_File &f, byte d);

// ============================================================
// Legacy/manual scratch buffers
// ============================================================

/**
 * @brief Install caller-owned internal scratch buffers.
 * @ingroup noodle_public
 *
 * This legacy hook is optional because Noodle now allocates scratch buffers on
 * demand. When installed, these pointers are used as-is, never resized, and not
 * freed by noodle_temp_buffers_free().
 *
 * @param b1 Input scratch buffer for file-backed input planes or sequences.
 * @param b2 Accumulation scratch buffer for one pre-pooling output map.
 */
void noodle_setup_temp_buffers(void *b1, void *b2);

/**
 * @brief Install only the caller-owned accumulation scratch buffer.
 * @ingroup noodle_public
 *
 * Use this when a legacy raw-pointer path needs only temp buffer 2.
 *
 * @param b2 Accumulation scratch buffer for one pre-pooling output map.
 */
void noodle_setup_temp_buffers(void *b2);

/**
 * @brief Release automatically allocated internal scratch buffers.
 * @ingroup noodle_public
 *
 * External buffers installed with noodle_setup_temp_buffers() are detached but
 * not freed.
 */
void noodle_temp_buffers_free(void);

// ============================================================
// Simple utility I/O
// ============================================================

/**
 * @brief Allocate a raw byte buffer and return it as a float pointer.
 * @ingroup noodle_public
 *
 * This compatibility wrapper calls malloc() with @p size bytes.
 *
 * @param size Number of bytes to allocate.
 * @return Allocated pointer, or nullptr on allocation failure.
 */
float *noodle_create_buffer(uint16_t size);

/**
 * @brief Free a buffer allocated by noodle_create_buffer().
 * @ingroup noodle_public
 * @param buffer Buffer pointer. Passing nullptr is allowed.
 */
void noodle_delete_buffer(float *buffer);

/**
 * @brief Write a float array to a file.
 * @ingroup noodle_public
 * @param array Source array with @p n values.
 * @param fn Output file.
 * @param n Number of values to write.
 */
void noodle_array_to_file(float *array, const char *fn, uint16_t n);

/**
 * @brief Write an `n x n` byte grid to a file in row-major order.
 * @ingroup noodle_public
 * @param grid Source grid with `n * n` values.
 * @param fn Output file.
 * @param n Grid width and height.
 */
void noodle_grid_to_file(byte *grid, const char *fn, uint16_t n);

/**
 * @brief Write an `n x n` float grid to a file in row-major order.
 * @ingroup noodle_public
 * @param grid Source grid with `n * n` values.
 * @param fn Output file.
 * @param n Grid width and height.
 */
void noodle_grid_to_file(float *grid, const char *fn, uint16_t n);

/**
 * @brief Read a float array from a file.
 * @ingroup noodle_public
 * @param fn Input file.
 * @param buffer Destination array with room for @p K floats.
 * @param K Number of values to read.
 */
void noodle_array_from_file(const char *fn, float *buffer, uint16_t K);

/**
 * @brief Read a `K x K` grid into a byte buffer.
 * @ingroup noodle_public
 * @param fn Input file.
 * @param buffer Destination buffer with room for `K * K` byte values.
 * @param K Grid width and height.
 */
void noodle_grid_from_file(const char *fn, byte *buffer, uint16_t K);

/**
 * @brief Read a `K x K` grid into an int8 buffer.
 * @ingroup noodle_public
 * @param fn Input file.
 * @param buffer Destination buffer with room for `K * K` int8 values.
 * @param K Grid width and height.
 */
void noodle_grid_from_file(const char *fn, int8_t *buffer, uint16_t K);

/**
 * @brief Read a `K x K` grid into a float buffer.
 * @ingroup noodle_public
 * @param fn Input file.
 * @param buffer Destination buffer with room for `K * K` float values.
 * @param K Grid width and height.
 */
void noodle_grid_from_file(const char *fn, float *buffer, uint16_t K);

// ============================================================
// Public file-backed layer API
// ============================================================

/**
 * @name File-backed layers
 *
 * File-backed convolution APIs read packed input tensors from files, stream
 * parameters from files or memory/PROGMEM, and write packed output tensors to a
 * file. They return the output width/length, or 0 on allocation/open/shape
 * failure.
 */

/**
 * @brief Run file-to-file 2D convolution on byte input feature maps.
 * @ingroup noodle_public
 *
 * Input is packed `[I][W][W]`; output is packed `[O][Vout][Vout]`.
 *
 * @param in_fn Input file.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param out_fn Output file.
 * @param W Input width and height.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters applied after bias and activation.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
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
 * Input is packed `[I][W][W]`; output is packed `[O][Vout][Vout]`.
 *
 * @param in_fn Input file.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param out_fn Output file.
 * @param W Input width and height.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters applied after bias and activation.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
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
 * @brief Run file-to-file 2D convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * `conv.weight` uses `[O][I][K][K]`; nullptr bias means zero bias.
 *
 * @param in_fn Input file with packed `[I][W][W]` planes.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param out_fn Output file for packed `[O][Vout][Vout]` planes.
 * @param W Input width and height.
 * @param conv Memory-backed convolution parameters.
 * @param pool Pooling parameters applied after bias and activation.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const ConvMem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run file-to-file 2D convolution with near-PROGMEM parameters.
 * @ingroup noodle_public
 *
 * `conv.weight` uses `[O][I][K][K]`; nullptr bias means zero bias.
 *
 * @param in_fn Input file with packed `[I][W][W]` planes.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param out_fn Output file for packed `[O][Vout][Vout]` planes.
 * @param W Input width and height.
 * @param conv Near-PROGMEM convolution parameters.
 * @param pool Pooling parameters applied after bias and activation.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const ConvProgmem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run file-to-file 1D convolution with file-backed parameters and pooling.
 * @ingroup noodle_public
 *
 * Input is packed `[I][W]`; output is packed `[O][Vout]`.
 *
 * @param in_fn Input file.
 * @param n_inputs Number of input channels.
 * @param out_fn Output file.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters applied after bias and activation.
 * @param progress_cb Optional progress callback.
 * @return Output length after pooling, or 0 on failure.
 */
uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const Conv &conv,
                       const Pool &pool,
                       CBFPtr progress_cb = NULL);

/**
 * @brief Run file-to-file 1D convolution with file-backed parameters.
 * @ingroup noodle_public
 *
 * This overload does not apply pooling.
 *
 * @param in_fn Input file with packed `[I][W]` sequences.
 * @param n_inputs Number of input channels.
 * @param out_fn Output file for packed `[O][V]` sequences.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv File-backed convolution parameters.
 * @param progress_cb Optional progress callback.
 * @return Output length before pooling, or 0 on failure.
 */
uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const Conv &conv,
                       CBFPtr progress_cb = NULL);

/**
 * @brief Run file-to-file 1D convolution with memory-backed parameters.
 * @ingroup noodle_public
 *
 * This overload does not apply pooling. `conv.weight` uses `[O][I][K]`;
 * nullptr bias means zero bias.
 *
 * @param in_fn Input file with packed `[I][W]` sequences.
 * @param n_inputs Number of input channels.
 * @param out_fn Output file for packed `[O][V]` sequences.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv Memory-backed convolution parameters.
 * @param progress_cb Optional progress callback.
 * @return Output length before pooling, or 0 on failure.
 */
uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb = NULL);

/**
 * @brief Run file-to-file depthwise 2D convolution with file-backed parameters.
 * @ingroup noodle_public
 *
 * Input and output are packed channel-first. `conv.weight_fn` stores
 * `[C][K][K]`, and `conv.bias_fn` stores one scalar per channel.
 *
 * @param in_fn Input file with packed `[C][W][W]` planes.
 * @param n_channels Number of channels.
 * @param out_fn Output file for packed `[C][Vout][Vout]` planes.
 * @param W Input width and height.
 * @param conv File-backed depthwise parameters.
 * @param pool Pooling parameters applied after bias and activation.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_dwconv_float(const char *in_fn,
                             uint16_t n_channels,
                             const char *out_fn,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb = NULL);

/**
 * @brief Run file-to-file depthwise 2D convolution with near-PROGMEM parameters.
 * @ingroup noodle_public
 *
 * `conv.weight` stores `[C][K][K]`, and nullptr bias means zero bias.
 *
 * @param in_fn Input file with packed `[C][W][W]` planes.
 * @param n_channels Number of channels.
 * @param out_fn Output file for packed `[C][Vout][Vout]` planes.
 * @param W Input width and height.
 * @param conv Near-PROGMEM depthwise parameters.
 * @param pool Pooling parameters applied after bias and activation.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_dwconv_float(const char *in_fn,
                             uint16_t n_channels,
                             const char *out_fn,
                             uint16_t W,
                             const ConvProgmem &conv,
                             const Pool &pool,
                             CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer from int8 memory to a file.
 * @ingroup noodle_public
 *
 * Input is a flat vector. Weights are read in `[O][I]` order from @p fcn.
 *
 * @param input Input vector with @p n_inputs values.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param out_fn Output file for @p n_outputs values.
 * @param fcn File-backed FCN parameters.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs, or 0 on failure.
 */
uint16_t noodle_fcn(const int8_t *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer from byte memory to a file.
 * @ingroup noodle_public
 *
 * Input is a flat vector. Weights are read in `[O][I]` order from @p fcn.
 *
 * @param input Input vector with @p n_inputs values.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param out_fn Output file for @p n_outputs values.
 * @param fcn File-backed FCN parameters.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs, or 0 on failure.
 */
uint16_t noodle_fcn(const byte *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer from a file to a file.
 * @ingroup noodle_public
 *
 * Input and output are flat vectors. Weights are read in `[O][I]` order from
 * @p fcn.
 *
 * @param in_fn Input file with @p n_inputs values.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param out_fn Output file for @p n_outputs values.
 * @param fcn File-backed FCN parameters.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs, or 0 on failure.
 */
uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);


// ============================================================
// Public NoodleTensor layer API
// ============================================================

/**
 * @name NoodleTensor layers
 *
 * These wrappers keep the existing raw kernels untouched. They read input
 * channel and width metadata from NoodleTensor, grow the output tensor as
 * needed, call the existing NoodleBuffer/raw layer, and update output shape.
 *
 * For Conv/ConvMem/ConvProgmem and FCN/FCNFile/FCNMem/FCNProgmem, set the
 * optional `.O` field before calling these wrappers. The progress callback is
 * intentionally not exposed here; use the older NoodleBuffer/raw overloads for
 * progress reporting or special instrumentation.
 */

/**
 * @brief Run 2D convolution on NoodleTensor input using file-backed parameters.
 * @ingroup noodle_public
 *
 * Input must be a rank-2 packed `[I][W][W]` tensor. @p conv must provide a
 * nonzero `O`; on success @p output becomes rank-2 `[O][Wout][Wout]`.
 *
 * @param input Rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters.
 * @return Output width after pooling, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_conv2d(NoodleTensor *input,
                       NoodleTensor *output,
                       const Conv &conv,
                       const Pool &pool);

/**
 * @brief Run 2D convolution on NoodleTensor input using memory-backed parameters.
 * @ingroup noodle_public
 *
 * Input must be a rank-2 packed `[I][W][W]` tensor. @p conv must provide a
 * nonzero `O`; on success @p output becomes rank-2 `[O][Wout][Wout]`.
 *
 * @param input Rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param conv Memory-backed convolution parameters.
 * @param pool Pooling parameters.
 * @return Output width after pooling, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_conv2d(NoodleTensor *input,
                       NoodleTensor *output,
                       const ConvMem &conv,
                       const Pool &pool);

/**
 * @brief Run 2D convolution on NoodleTensor input using near-PROGMEM parameters.
 * @ingroup noodle_public
 *
 * Input must be a rank-2 packed `[I][W][W]` tensor. @p conv must provide a
 * nonzero `O`; on success @p output becomes rank-2 `[O][Wout][Wout]`.
 *
 * @param input Rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param conv Near-PROGMEM convolution parameters.
 * @param pool Pooling parameters.
 * @return Output width after pooling, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_conv2d(NoodleTensor *input,
                       NoodleTensor *output,
                       const ConvProgmem &conv,
                       const Pool &pool);

/**
 * @brief Run 2D transpose convolution on NoodleTensor input.
 * @ingroup noodle_public
 *
 * Input must be a rank-2 packed `[I][W][W]` tensor. @p conv must provide a
 * nonzero `O`; on success @p output becomes rank-2 `[O][Wout][Wout]`.
 *
 * @param input Rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param conv Memory-backed transpose convolution parameters.
 * @return Output width, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_conv_transpose2d(NoodleTensor *input,
                                 NoodleTensor *output,
                                 const ConvMem &conv);

/**
 * @brief Run 1D convolution on NoodleTensor input without pooling.
 * @ingroup noodle_public
 *
 * Input must be a rank-1 packed `[I][W]` tensor. @p conv must provide a nonzero
 * `O`; on success @p output becomes rank-1 `[O][Wout]`.
 *
 * @param input Rank-1 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param conv Memory-backed 1D convolution parameters.
 * @return Output length before pooling, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_conv1d(NoodleTensor *input,
                       NoodleTensor *output,
                       const ConvMem &conv);

/**
 * @brief Run 1D convolution on NoodleTensor input with pooling.
 * @ingroup noodle_public
 *
 * Input must be a rank-1 packed `[I][W]` tensor. @p conv must provide a nonzero
 * `O`; on success @p output becomes rank-1 `[O][Wout]`.
 *
 * @param input Rank-1 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param conv Memory-backed 1D convolution parameters.
 * @param pool Pooling parameters applied after bias and activation.
 * @return Output length after pooling, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_conv1d(NoodleTensor *input,
                       NoodleTensor *output,
                       const ConvMem &conv,
                       const Pool &pool);

/**
 * @brief Run depthwise 2D convolution on NoodleTensor input using file parameters.
 * @ingroup noodle_public
 *
 * Input must be a rank-2 packed `[C][W][W]` tensor. On success @p output keeps
 * the same channel count and becomes rank-2 `[C][Wout][Wout]`.
 *
 * @param input Rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param conv File-backed depthwise parameters.
 * @param pool Pooling parameters.
 * @return Output width after pooling, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_dwconv2d(NoodleTensor *input,
                         NoodleTensor *output,
                         const Conv &conv,
                         const Pool &pool);

/**
 * @brief Run depthwise 2D convolution on NoodleTensor input using memory parameters.
 * @ingroup noodle_public
 *
 * Input must be a rank-2 packed `[C][W][W]` tensor. On success @p output keeps
 * the same channel count and becomes rank-2 `[C][Wout][Wout]`.
 *
 * @param input Rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param conv Memory-backed depthwise parameters.
 * @param pool Pooling parameters.
 * @return Output width after pooling, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_dwconv2d(NoodleTensor *input,
                         NoodleTensor *output,
                         const ConvMem &conv,
                         const Pool &pool);

/**
 * @brief Run depthwise 2D convolution on NoodleTensor input using PROGMEM parameters.
 * @ingroup noodle_public
 *
 * Input must be a rank-2 packed `[C][W][W]` tensor. On success @p output keeps
 * the same channel count and becomes rank-2 `[C][Wout][Wout]`.
 *
 * @param input Rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param conv Near-PROGMEM depthwise parameters.
 * @param pool Pooling parameters.
 * @return Output width after pooling, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_dwconv2d(NoodleTensor *input,
                         NoodleTensor *output,
                         const ConvProgmem &conv,
                         const Pool &pool);

/**
 * @brief Apply 2D pooling to a rank-2 NoodleTensor.
 * @ingroup noodle_public
 *
 * Input must be packed `[C][W][W]`. On success @p output becomes rank-2
 * `[C][Wout][Wout]` and uses the compile-time `NOODLE_POOL_MODE`.
 *
 * @param input Rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param K Pool window size.
 * @param S Pool stride.
 * @return Output width, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_pool2d(NoodleTensor *input,
                       NoodleTensor *output,
                       uint16_t K,
                       uint16_t S);

/**
 * @brief Apply global average pooling in place to a rank-2 NoodleTensor.
 * @ingroup noodle_public
 *
 * On success the tensor changes from packed `[C][W][W]` to rank-1 `[C][1]`.
 *
 * @param inout Rank-2 tensor updated in place.
 * @return Channel count, or 0 on invalid input.
 */
uint16_t noodle_gap(NoodleTensor *inout);

/**
 * @brief Reduce each channel in place with the NoodleBuffer max-pooling helper.
 * @ingroup noodle_public
 *
 * On success the tensor changes from rank-2 channel data to rank-1 `[C][1]`.
 *
 * @param inout Rank-2 tensor updated in place.
 * @return Channel count, or 0 on invalid input.
 */
uint16_t noodle_gmp(NoodleTensor *inout);

/**
 * @brief Flatten a rank-2 NoodleTensor into a rank-1 vector.
 * @ingroup noodle_public
 *
 * Reads packed `[C][W][W]`, writes HWC-like flat order into @p output, and marks
 * @p output as rank-1 `[N][1]`.
 *
 * @param input Rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @return Number of floats written, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_flat(NoodleTensor *input, NoodleTensor *output);

/**
 * @brief Concatenate two rank-2 NoodleTensors by channel.
 * @ingroup noodle_public
 *
 * Inputs must have the same width. On success @p output becomes rank-2
 * `[(A->C + B->C)][W][W]`.
 *
 * @param A First rank-2 input tensor.
 * @param B Second rank-2 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @return Combined channel count, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_concat(NoodleTensor *A, NoodleTensor *B, NoodleTensor *output);

/**
 * @brief Run a fully connected layer on a rank-1 NoodleTensor.
 * @ingroup noodle_public
 *
 * @p fcn must provide a nonzero `O`; on success @p output becomes rank-1
 * `[O][1]`.
 *
 * @param input Rank-1 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param fcn Memory-backed FCN parameters.
 * @return Output count, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_fcn(NoodleTensor *input,
                    NoodleTensor *output,
                    const FCNMem &fcn);

/**
 * @brief Run a fully connected layer on a rank-1 NoodleTensor from files.
 * @ingroup noodle_public
 *
 * @p fcn must provide a nonzero `O`; on success @p output becomes rank-1
 * `[O][1]`.
 *
 * @param input Rank-1 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param fcn File-backed FCN parameters.
 * @return Output count, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_fcn(NoodleTensor *input,
                    NoodleTensor *output,
                    const FCNFile &fcn);

/**
 * @brief Run a fully connected layer on a rank-1 NoodleTensor from far PROGMEM.
 * @ingroup noodle_public
 *
 * @p fcn must provide a nonzero `O`; on success @p output becomes rank-1
 * `[O][1]`.
 *
 * @param input Rank-1 input tensor.
 * @param output Output tensor grown and reshaped on success.
 * @param fcn Far-PROGMEM FCN parameters.
 * @return Output count on AVR, or 0 on invalid input/allocation failure/non-AVR.
 */
uint16_t noodle_fcn(NoodleTensor *input,
                    NoodleTensor *output,
                    const FCNProgmem &fcn);

/**
 * @brief Apply softmax in place over a rank-1 NoodleTensor.
 * @ingroup noodle_public
 *
 * @param input_output Rank-1 tensor updated in place.
 * @return Number of vector elements, or 0 on invalid input.
 */
uint16_t noodle_soft_max(NoodleTensor *input_output);

/**
 * @brief Apply sigmoid in place over a NoodleTensor's logical elements.
 * @ingroup noodle_public
 *
 * The tensor shape is unchanged.
 *
 * @param input_output Tensor updated in place.
 * @return Number of logical elements, or 0 on invalid input.
 */
uint16_t noodle_sigmoid(NoodleTensor *input_output);

/**
 * @brief Apply ReLU in place over a NoodleTensor's logical elements.
 * @ingroup noodle_public
 *
 * The tensor shape is unchanged.
 *
 * @param input_output Tensor updated in place.
 * @return Number of logical elements, or 0 on invalid input.
 */
uint16_t noodle_relu(NoodleTensor *input_output);

// ============================================================
// Public NoodleBuffer RAM-to-RAM layer API
// ============================================================

/**
 * @name NoodleBuffer RAM-to-RAM layers
 *
 * These overloads read input from `input->data`, grow @p output with
 * noodle_buffer_require(), and write the result into `output->data`. They return
 * the output width/length or output count, and return 0 on null input, null
 * storage, invalid shape, allocation failure, or a failing lower-level layer.
 */

/**
 * @brief Run 2D convolution using file-backed parameters.
 * @ingroup noodle_public
 *
 * Input is packed `[I][W][W]`; output is packed `[O][Vout][Vout]`.
 *
 * @param input Input NoodleBuffer with packed feature maps.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param output Output NoodleBuffer grown as needed.
 * @param W Input width and height.
 * @param conv File-backed convolution parameters.
 * @param pool Pooling parameters.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_conv_float(NoodleBuffer *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           NoodleBuffer *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run 2D convolution using memory-backed parameters.
 * @ingroup noodle_public
 *
 * `conv.weight` uses `[O][I][K][K]`; nullptr bias means zero bias.
 *
 * @param input Input NoodleBuffer with packed `[I][W][W]` feature maps.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param output Output NoodleBuffer grown as needed.
 * @param W Input width and height.
 * @param conv Memory-backed convolution parameters.
 * @param pool Pooling parameters.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_conv_float(NoodleBuffer *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           NoodleBuffer *output,
                           uint16_t W,
                           const ConvMem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run 2D convolution using near-PROGMEM parameters.
 * @ingroup noodle_public
 *
 * `conv.weight` uses `[O][I][K][K]`; nullptr bias means zero bias.
 *
 * @param input Input NoodleBuffer with packed `[I][W][W]` feature maps.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param output Output NoodleBuffer grown as needed.
 * @param W Input width and height.
 * @param conv Near-PROGMEM convolution parameters.
 * @param pool Pooling parameters.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_conv_float(NoodleBuffer *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           NoodleBuffer *output,
                           uint16_t W,
                           const ConvProgmem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/**
 * @brief Run 1D convolution using memory-backed parameters.
 * @ingroup noodle_public
 *
 * This overload does not apply pooling. Input is `[I][W]`; output is `[O][V]`.
 *
 * @param input Input NoodleBuffer with packed 1D feature maps.
 * @param n_inputs Number of input channels.
 * @param output Output NoodleBuffer grown as needed.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv Memory-backed 1D convolution parameters.
 * @param progress_cb Optional progress callback.
 * @return Output length before pooling, or 0 on failure.
 */
uint16_t noodle_conv1d(NoodleBuffer *input,
                       uint16_t n_inputs,
                       NoodleBuffer *output,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb = NULL);

/**
 * @brief Run 1D convolution using memory-backed parameters and pooling.
 * @ingroup noodle_public
 *
 * Input is `[I][W]`; output is `[O][Vout]`.
 *
 * @param input Input NoodleBuffer with packed 1D feature maps.
 * @param n_inputs Number of input channels.
 * @param output Output NoodleBuffer grown as needed.
 * @param n_outputs Number of output channels.
 * @param W Input sequence length.
 * @param conv Memory-backed 1D convolution parameters.
 * @param pool Pooling parameters applied after bias and activation.
 * @param progress_cb Optional progress callback.
 * @return Output length after pooling, or 0 on failure.
 */
uint16_t noodle_conv1d(NoodleBuffer *input,
                       uint16_t n_inputs,
                       NoodleBuffer *output,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       const Pool &pool,
                       CBFPtr progress_cb = NULL);

/**
 * @brief Run depthwise 2D convolution using file-backed parameters.
 * @ingroup noodle_public
 *
 * Input and output are packed channel-first. `conv.weight_fn` stores
 * `[C][K][K]`.
 *
 * @param input Input NoodleBuffer with packed `[C][W][W]` planes.
 * @param n_channels Number of channels.
 * @param output Output NoodleBuffer grown as needed.
 * @param W Input width and height.
 * @param conv File-backed depthwise parameters.
 * @param pool Pooling parameters.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_dwconv_float(NoodleBuffer *input,
                             uint16_t n_channels,
                             NoodleBuffer *output,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb = NULL);

/**
 * @brief Run depthwise 2D convolution using memory-backed parameters.
 * @ingroup noodle_public
 *
 * `conv.weight` stores `[C][K][K]`; nullptr bias means zero bias.
 *
 * @param input Input NoodleBuffer with packed `[C][W][W]` planes.
 * @param n_channels Number of channels.
 * @param output Output NoodleBuffer grown as needed.
 * @param W Input width and height.
 * @param conv Memory-backed depthwise parameters.
 * @param pool Pooling parameters.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_dwconv_float(NoodleBuffer *input,
                             uint16_t n_channels,
                             NoodleBuffer *output,
                             uint16_t W,
                             const ConvMem &conv,
                             const Pool &pool,
                             CBFPtr progress_cb = NULL);

/**
 * @brief Run depthwise 2D convolution using near-PROGMEM parameters.
 * @ingroup noodle_public
 *
 * `conv.weight` stores `[C][K][K]`; nullptr bias means zero bias.
 *
 * @param input Input NoodleBuffer with packed `[C][W][W]` planes.
 * @param n_channels Number of channels.
 * @param output Output NoodleBuffer grown as needed.
 * @param W Input width and height.
 * @param conv Near-PROGMEM depthwise parameters.
 * @param pool Pooling parameters.
 * @param progress_cb Optional progress callback.
 * @return Output width after pooling, or 0 on failure.
 */
uint16_t noodle_dwconv_float(NoodleBuffer *input,
                             uint16_t n_channels,
                             NoodleBuffer *output,
                             uint16_t W,
                             const ConvProgmem &conv,
                             const Pool &pool,
                             CBFPtr progress_cb = NULL);

/**
 * @brief Run 2D transpose convolution using memory-backed parameters.
 * @ingroup noodle_public
 *
 * Input is packed `[I][W][W]`; output is packed `[O][Vt][Vt]`. Weights use
 * `[O][I][K][K]`.
 *
 * For explicit padding, callers set `conv.OP` so
 * `Vt = (W - 1) * conv.S - 2 * conv.P + conv.K + conv.OP` matches the desired
 * output width. With `conv.P == 65535`, SAME-style output uses `Vt = W * conv.S`.
 *
 * @param input Input NoodleBuffer with packed feature maps.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param output Output NoodleBuffer grown as needed.
 * @param W Input width and height.
 * @param conv Memory-backed transpose convolution parameters.
 * @param progress_cb Optional progress callback.
 * @return Output width, or 0 on failure.
 */
uint16_t noodle_conv_transpose_float(NoodleBuffer *input,
                                     uint16_t n_inputs,
                                     uint16_t n_outputs,
                                     NoodleBuffer *output,
                                     uint16_t W,
                                     const ConvMem &conv,
                                     CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer using memory-backed parameters.
 * @ingroup noodle_public
 *
 * @param input Input NoodleBuffer containing a flat vector.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param output Output NoodleBuffer grown to @p n_outputs floats.
 * @param fcn Memory-backed FCN parameters.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs, or 0 on failure.
 */
uint16_t noodle_fcn(NoodleBuffer *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    NoodleBuffer *output,
                    const FCNMem &fcn,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer using file-backed parameters.
 * @ingroup noodle_public
 *
 * @param input Input NoodleBuffer containing a flat vector.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param output Output NoodleBuffer grown to @p n_outputs floats.
 * @param fcn File-backed FCN parameters.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs, or 0 on failure.
 */
uint16_t noodle_fcn(NoodleBuffer *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    NoodleBuffer *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer using far-PROGMEM parameters.
 * @ingroup noodle_public
 *
 * @param input Input NoodleBuffer containing a flat vector.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param output Output NoodleBuffer grown to @p n_outputs floats.
 * @param fcn Far-PROGMEM FCN parameters.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs on AVR, or 0 on failure/non-AVR.
 */
uint16_t noodle_fcn(NoodleBuffer *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    NoodleBuffer *output,
                    const FCNProgmem &fcn,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer from byte memory to a NoodleBuffer.
 * @ingroup noodle_public
 * @param input Input vector with @p n_inputs values.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param output Output NoodleBuffer grown to @p n_outputs floats.
 * @param fcn File-backed FCN parameters.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs, or 0 on failure.
 */
uint16_t noodle_fcn(const byte *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    NoodleBuffer *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer from int8 memory to a NoodleBuffer.
 * @ingroup noodle_public
 * @param input Input vector with @p n_inputs values.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param output Output NoodleBuffer grown to @p n_outputs floats.
 * @param fcn File-backed FCN parameters.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs, or 0 on failure.
 */
uint16_t noodle_fcn(const int8_t *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    NoodleBuffer *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer from a file to a NoodleBuffer.
 * @ingroup noodle_public
 * @param in_fn Input file with @p n_inputs values.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param output Output NoodleBuffer grown to @p n_outputs floats.
 * @param fcn File-backed FCN parameters.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs, or 0 on failure.
 */
uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    NoodleBuffer *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);

/**
 * @brief Run a fully connected layer using near-PROGMEM weights.
 * @ingroup noodle_public
 *
 * `weight` is read with noodle_pgm_float() in `[O][I]` order. `bias` may be
 * nullptr for zero bias.
 *
 * @param input Input NoodleBuffer containing a flat vector.
 * @param n_inputs Input vector length.
 * @param n_outputs Number of output neurons.
 * @param output Output NoodleBuffer grown to @p n_outputs floats.
 * @param weight Near-PROGMEM row-major `[O][I]` weights.
 * @param bias Near-PROGMEM output biases, or nullptr.
 * @param act Activation applied after each output.
 * @param progress_cb Optional progress callback.
 * @return @p n_outputs, or 0 on failure.
 */
uint16_t noodle_fcn_progmem(NoodleBuffer *input,
                            uint16_t n_inputs,
                            uint16_t n_outputs,
                            NoodleBuffer *output,
                            const float *weight,
                            const float *bias,
                            Activation act,
                            CBFPtr progress_cb = NULL);

// ============================================================
// Tensor utilities and activations
// ============================================================

/**
 * @brief Flatten a packed file tensor into a NoodleBuffer.
 * @ingroup noodle_public
 *
 * Reads packed `[C][V][V]` input and writes HWC-like order:
 * `output[pixel * C + channel]`. The output buffer grows automatically to
 * `V * V * n_filters` floats.
 *
 * @param in_fn Input file containing packed channel-first planes.
 * @param output Destination buffer grown as needed.
 * @param V Input plane width and height.
 * @param n_filters Number of channel planes.
 * @return Number of floats written, or 0 on null input/allocation failure.
 */
uint16_t noodle_flat(const char *in_fn,
                     NoodleBuffer *output,
                     uint16_t V,
                     uint16_t n_filters);

/**
 * @brief Flatten a packed NoodleBuffer tensor into HWC-like order.
 * @ingroup noodle_public
 *
 * Reads `input` as packed `[C][V][V]`, grows `output` to
 * `V * V * n_filters` floats, and writes `output[pixel * C + channel]`.
 *
 * @param input Source buffer with packed channel-first planes.
 * @param output Destination buffer grown as needed.
 * @param V Input plane width and height.
 * @param n_filters Number of channel planes.
 * @return Number of floats written, or 0 on null input/allocation failure.
 */
uint16_t noodle_flat(NoodleBuffer *input,
                     NoodleBuffer *output,
                     uint16_t V,
                     uint16_t n_filters);

/**
 * @brief Convert HWC-like NoodleBuffer data to packed channel-first data.
 * @ingroup noodle_public
 *
 * The destination buffer grows automatically to `W * W * C` floats.
 *
 * @param src_hwc Source buffer in `src[pixel * C + channel]` order.
 * @param dst_chw Destination buffer grown as needed.
 * @param W Output plane width and height.
 * @param C Number of channel planes.
 * @return Number of floats written, or 0 on null input/allocation failure.
 */
uint16_t noodle_reshape(NoodleBuffer *src_hwc,
                        NoodleBuffer *dst_chw,
                        uint16_t W,
                        uint16_t C);

/**
 * @brief Apply global average pooling in place on packed channel-first maps.
 * @ingroup noodle_public
 *
 * Reduces `[C][W][W]` to `[C]` by writing each channel mean into the first
 * `C` positions of @p inout.
 *
 * @param inout Buffer containing packed `[C][W][W]` data.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @return @p C, or 0 when @p inout has no data.
 */
uint16_t noodle_gap(NoodleBuffer *inout, uint16_t C, uint16_t W);

/**
 * @brief Apply global max pooling in place on packed channel-first data.
 * @ingroup noodle_public
 *
 * Reduces each channel to one maximum value in the first `C` positions of
 * @p inout. The current helper scans @p W values per channel.
 *
 * @param inout Buffer containing packed channel data.
 * @param C Number of channels.
 * @param W Number of values scanned per channel.
 * @return @p C, or 0 when @p inout has no data.
 */
uint16_t noodle_gmp(NoodleBuffer *inout, uint16_t C, uint16_t W);

/**
 * @brief Concatenate two packed channel-first tensors by channel.
 * @ingroup noodle_public
 *
 * Reads @p A as `[C_A][V][V]` and @p B as `[C_B][V][V]`, grows @p output to
 * `(C_A + C_B) * V * V` floats, then copies all @p A channels followed by all
 * @p B channels.
 *
 * @param A First input buffer.
 * @param C_A Number of channels in @p A.
 * @param B Second input buffer.
 * @param C_B Number of channels in @p B.
 * @param output Destination buffer grown as needed.
 * @param V Input and output plane width and height.
 * @return Combined channel count, or 0 on null input/allocation failure.
 */
uint16_t noodle_concat(NoodleBuffer *A, uint16_t C_A,
                       NoodleBuffer *B, uint16_t C_B,
                       NoodleBuffer *output, uint16_t V);

/**
 * @brief Apply 2D pooling to a packed channel-first tensor.
 * @ingroup noodle_public
 *
 * Reads @p input as `[C][W][W]`, grows @p output for `[C][Wo][Wo]`, and pools
 * each channel independently using the compile-time `NOODLE_POOL_MODE`. When
 * `NOODLE_POOL_MODE == NOODLE_POOL_NONE`, the helper copies each plane and
 * returns @p W. The wrapper rejects in-place pooling because overlapping
 * windows can overwrite values that are still needed.
 *
 * @param input Source tensor buffer.
 * @param C Number of channels.
 * @param W Input plane width and height.
 * @param output Destination tensor buffer grown as needed.
 * @param K Pool window size.
 * @param S Pool stride.
 * @return Output plane width, or 0 on invalid input/allocation failure.
 */
uint16_t noodle_pool2d(NoodleBuffer *input,
                       uint16_t C,
                       uint16_t W,
                       NoodleBuffer *output,
                       uint16_t K,
                       uint16_t S);

/**
 * @brief Apply numerically stabilized softmax in place on a NoodleBuffer.
 * @ingroup noodle_public
 * @param input_output Vector buffer updated in place.
 * @param n Number of vector elements.
 * @return @p n, or 0 when @p input_output has no data.
 */
uint16_t noodle_soft_max(NoodleBuffer *input_output, uint16_t n);

/**
 * @brief Apply sigmoid in place on a NoodleBuffer.
 * @ingroup noodle_public
 * @param input_output Vector buffer updated in place.
 * @param n Number of vector elements.
 * @return @p n, or 0 when @p input_output has no data.
 */
uint16_t noodle_sigmoid(NoodleBuffer *input_output, uint16_t n);

/**
 * @brief Compute sigmoid for one scalar.
 * @ingroup noodle_public
 * @param x Scalar input.
 * @return Logistic sigmoid of @p x.
 */
float noodle_sigmoidf(float x);

/**
 * @brief Apply logistic sigmoid in place on a NoodleBuffer.
 * @ingroup noodle_public
 * @param input_output Vector buffer updated in place.
 * @param n Number of vector elements.
 * @return @p n, or 0 when @p input_output has no data.
 */
uint16_t noodle_logit(NoodleBuffer *input_output, uint16_t n);

/**
 * @brief Apply ReLU in place on a NoodleBuffer.
 * @ingroup noodle_public
 * @param input_output Vector buffer updated in place.
 * @param n Number of vector elements.
 * @return @p n, or 0 when @p input_output has no data.
 */
uint16_t noodle_relu(NoodleBuffer *input_output, uint16_t n);

/**
 * @brief Find the maximum value and its index in a NoodleBuffer vector.
 * @ingroup noodle_public
 * @param input Input vector buffer.
 * @param n Number of vector elements to inspect.
 * @param max_val Receives the maximum value, or 0.0 for null/empty input.
 * @param max_idx Receives the maximum index, or 0 for null/empty input.
 */
void noodle_find_max(NoodleBuffer *input, uint16_t n,
                     float &max_val, uint16_t &max_idx);

/**
 * @brief Apply 1D batch normalization in place to a NoodleBuffer vector.
 * @ingroup noodle_public
 *
 * `gamma`, `beta`, `mean`, and `var` each contain @p N values.
 *
 * @param x Vector buffer updated in place.
 * @param N Number of vector elements.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p N, or 0 when @p x has no data.
 */
uint16_t noodle_bn1d(NoodleBuffer *x, uint16_t N,
                     const float *gamma, const float *beta,
                     const float *mean, const float *var, float eps);

/**
 * @brief Apply 1D batch normalization from a packed parameter array.
 * @ingroup noodle_public
 *
 * `bn_params` is packed as `[gamma[N]][beta[N]][mean[N]][var[N]]`.
 *
 * @param x Vector buffer updated in place.
 * @param N Number of vector elements.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p N, or 0 when @p x has no data.
 */
uint16_t noodle_bn1d(NoodleBuffer *x, uint16_t N,
                     const float *bn_params, float eps);

/**
 * @brief Apply 1D batch normalization followed by ReLU in place.
 * @ingroup noodle_public
 *
 * `gamma`, `beta`, `mean`, and `var` each contain @p N values.
 *
 * @param x Vector buffer updated in place.
 * @param N Number of vector elements.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p N, or 0 when @p x has no data.
 */
uint16_t noodle_bn1d_relu(NoodleBuffer *x, uint16_t N,
                          const float *gamma, const float *beta,
                          const float *mean, const float *var, float eps);

/**
 * @brief Apply packed 1D batch normalization followed by ReLU in place.
 * @ingroup noodle_public
 *
 * `bn_params` is packed as `[gamma[N]][beta[N]][mean[N]][var[N]]`.
 *
 * @param x Vector buffer updated in place.
 * @param N Number of vector elements.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p N, or 0 when @p x has no data.
 */
uint16_t noodle_bn1d_relu(NoodleBuffer *x, uint16_t N,
                          const float *bn_params, float eps);

/**
 * @brief Apply 2D channel-wise batch normalization in place.
 * @ingroup noodle_public
 *
 * Treats @p x as packed `[C][W][W]`. The parameter arrays each contain @p C
 * values, one per channel.
 *
 * @param x Tensor buffer updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W, or 0 when @p x has no data.
 */
uint16_t noodle_bn2d(NoodleBuffer *x, uint16_t C, uint16_t W,
                     const float *gamma, const float *beta,
                     const float *mean, const float *var, float eps);

/**
 * @brief Apply 2D channel-wise batch normalization from packed parameters.
 * @ingroup noodle_public
 *
 * Treats @p x as packed `[C][W][W]`. `bn_params` is packed as
 * `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor buffer updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W, or 0 when @p x has no data.
 */
uint16_t noodle_bn2d(NoodleBuffer *x, uint16_t C, uint16_t W,
                     const float *bn_params, float eps);

/**
 * @brief Apply 2D channel-wise batch normalization followed by ReLU.
 * @ingroup noodle_public
 *
 * Treats @p x as packed `[C][W][W]`. The parameter arrays each contain @p C
 * values, one per channel.
 *
 * @param x Tensor buffer updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W, or 0 when @p x has no data.
 */
uint16_t noodle_bn2d_relu(NoodleBuffer *x, uint16_t C, uint16_t W,
                          const float *gamma, const float *beta,
                          const float *mean, const float *var, float eps);

/**
 * @brief Apply packed 2D batch normalization followed by ReLU.
 * @ingroup noodle_public
 *
 * Treats @p x as packed `[C][W][W]`. `bn_params` is packed as
 * `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor buffer updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W, or 0 when @p x has no data.
 */
uint16_t noodle_bn2d_relu(NoodleBuffer *x, uint16_t C, uint16_t W,
                          const float *bn_params, float eps);

/**
 * @brief Backward-compatible alias for noodle_bn2d().
 * @ingroup noodle_public
 *
 * Treats @p x as packed `[C][W][W]`.
 *
 * @param x Tensor buffer updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W, or 0 when @p x has no data.
 */
uint16_t noodle_bn(NoodleBuffer *x, uint16_t C, uint16_t W,
                   const float *gamma, const float *beta,
                   const float *mean, const float *var, float eps);

/**
 * @brief Backward-compatible alias for packed-parameter noodle_bn2d().
 * @ingroup noodle_public
 *
 * Treats @p x as packed `[C][W][W]`. `bn_params` is packed as
 * `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor buffer updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W, or 0 when @p x has no data.
 */
uint16_t noodle_bn(NoodleBuffer *x, uint16_t C, uint16_t W,
                   const float *bn_params, float eps);

/**
 * @brief Backward-compatible alias for noodle_bn2d_relu().
 * @ingroup noodle_public
 *
 * Treats @p x as packed `[C][W][W]`.
 *
 * @param x Tensor buffer updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W, or 0 when @p x has no data.
 */
uint16_t noodle_bn_relu(NoodleBuffer *x, uint16_t C, uint16_t W,
                        const float *gamma, const float *beta,
                        const float *mean, const float *var, float eps);

/**
 * @brief Backward-compatible alias for packed-parameter noodle_bn2d_relu().
 * @ingroup noodle_public
 *
 * Treats @p x as packed `[C][W][W]`. `bn_params` is packed as
 * `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor buffer updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W, or 0 when @p x has no data.
 */
uint16_t noodle_bn_relu(NoodleBuffer *x, uint16_t C, uint16_t W,
                        const float *bn_params, float eps);
