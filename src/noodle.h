/**
 *  @file noodle.h
 *  @brief CNN/ML primitives for tiny MCUs with pluggable filesystem backends.
 *  @ingroup noodle_public
 *
 *  @details
 *  Noodle provides minimal convolution, pooling, flatten, fully-connected and activation
 *  routines that operate either entirely in memory or by streaming tensors/parameters to
 *  and from a filesystem (SD/FFat/SD_MMC). Backends are selected in @ref noodle_fs.h via
 *  preprocessor flags. The library is designed for extremely small RAM budgets, so most
 *  file-based APIs reuse two caller-supplied temporary buffers
 *  (see @ref noodle_setup_temp_buffers).
 *
 *  ### Backends
 *  Select exactly one backend flag before including this header: `NOODLE_USE_SDFAT`, `NOODLE_USE_SD_MMC`, `NOODLE_USE_FFAT`, or `NOODLE_USE_LITTLEFS`.
 *  See @ref noodle_fs.h for backend details and required platform libraries.
 *
 *  ### Temporary buffers
 *  Call ::noodle_setup_temp_buffers() once prior to convolution/FCN routines that read from files.
 *  Unless otherwise stated:
 *   - temp buffer #1: at least `W*W*sizeof(float)` bytes
 *   - temp buffer #2: at least `W*W*sizeof(float)` bytes (accumulator)
 *
 *  ### Tensor layout and file format
 *  File-based APIs use *packed CHW* by default: channel-major planes stored back-to-back.
 *  Each plane is stored in row-major order as whitespace-separated numeric tokens (ASCII).
 *
 *  - Packed 2D feature map file: `[ch0 W×W][ch1 W×W]...[ch(C-1) W×W]`
 *  - Packed 1D feature map file: `[ch0 W][ch1 W]...[ch(C-1) W]`
 *  - Weights file (Conv): ordered by output-major then input: `(O0,I0) kernel, (O0,I1) kernel, ...`
 *  - Bias file: one scalar per output channel (n_outputs values).
 *
 * # ## Convolution
 *  All convolutions in Noodle use symmetric padding.
 *
 * ### Pooling mode
 * 2D pooling mode is compile-time selectable via @ref NOODLE_POOL_MODE in @ref noodle_config.h
 * (MAX or MEAN).
 */

/** 
 *  @defgroup noodle_public Public Functions
 *  Public functions, types, and configuration intended for application use.
 *  @defgroup noodle_internal Internal helpers
 *  Internal helpers and implementation details. Not intended for direct application use.
 */
 
#pragma once

#include <stdint.h>
#ifdef ARDUINO
#include <Arduino.h>
#endif

#include "noodle_config.h"   // pooling mode & backend selection macros
#include "noodle_fs.h"       // NDL_File + NOODLE_FS wrappers

#ifndef ARDUINO
typedef unsigned char byte;    ///< Minimal Arduino-compatible alias when not building for Arduino.
#endif
// ============================================================================

/** 
 *  2D convolution activations.
 *  @ingroup noodle_public
 *  @enum Activation
 *  @brief Post-layer activation applied after bias (when supported).
 *
 * Notes:
 * - ACT_SOFTMAX is typically used only for the final classification layer/head.
 * - Most convolution routines support ACT_RELU and ACT_NONE; check each function's docs.
 */
enum Activation : uint8_t { ACT_NONE = 0, ACT_RELU = 1 , ACT_SOFTMAX = 2};

/** 
 * File-backed parameters 
 *  @ingroup noodle_public
 *  @brief File-backed convolution parameters.
 *
 * The weight and bias files are read sequentially:
 * - @p weight_fn stores kernels ordered by output-major then input-major.
 * - @p bias_fn stores one scalar bias per output channel.
 */
struct Conv {
  uint16_t K;        ///< Kernel size
  uint16_t P = 0;    ///< Padding
  uint16_t S = 1;    ///< Stride

  const char *weight_fn = nullptr;
  const char *bias_fn   = nullptr;

  Activation act = ACT_RELU;
};

/** 
 * Variable-backed parameters 
 *  @ingroup noodle_public
 *  @brief Memory-backed convolution parameters.
 *
 * The @p weight array is laid out as `(O * n_inputs + I) * (K*K)` contiguous floats (output-major).
 * The @p bias array stores @c n_outputs floats.
 */
struct ConvMem {
  uint16_t K;        ///< Kernel size
  uint16_t P = 0;    ///< Padding
  uint16_t S = 1;    ///< Stride

  const float *weight = nullptr;
  const float *bias   = nullptr;

  Activation act = ACT_RELU;
};

/** 
 *  @brief 2D pooling parameters. 
 *  Use M = 1 and T = 1 for identity (no pooling). 
 *  @ingroup noodle_public
 *  @brief Pooling parameters for 1D/2D pooling.
 *
 *  To disable pooling, use identity parameters (e.g., M=1 and T=1), or pass a Pool where your caller 
 *  logic skips pooling.
 */
struct Pool {
  uint16_t M = 1;    ///< Pool kernel
  uint16_t T = 1;    ///< Pool stride
};

/** 
 *  @brief Progress callback type used by long-running routines.
 *  @ingroup noodle_public
 *  @param progress A normalized progress in [0,1], monotonically nondecreasing.
 */
typedef void (*CBFPtr)(float progress);

/** 
 *  @name Common parameter semantics
 *
 *  Standard meanings shared by many functions.
 *  @param W   Input spatial width (2D) or length (1D).
 *  @param K   Kernel size (2D: KxK, 1D: K).
 *  @param S   Stride.
 *  @param P   Zero-padding (per side). 2D uses top/left padding of size @p P.
 *  @param M   Pool kernel size (2D: MxM).
 *  @param T   Pool stride.
 *  @param n_inputs  Number of input channels/features.
 *  @param n_outputs Number of output channels/features.
 *  @param in_fn  Base input filename template (see File naming convention).
 *  @param out_fn Base output filename template.
 *  @param weight_fn Weight filename template; receives both I and O indices.
 *  @param bias_fn   Bias filename (one bias per output channel, scalar per line).
 *  @param with_relu If true, apply ReLU after bias.
 */

/** 
 *  @brief Provide two reusable temporary buffers used internally by file-streaming operations.
 *  @ingroup noodle_public
 *  Must be called before conv/FCN variants that read from files.
 *  Two temp buffers are needed for operations that read from a file.
 *  For C*W*W tensor, the buffer should be W*W
 *  @param b1 Buffer #1 (input scratch). See size guidance above.
 *  @param b2 Buffer #2 (float accumulator). See size guidance above.
 */
void   noodle_setup_temp_buffers(void *b1, void *b2);

/** 
 *  @brief Provide a single reusable temporary buffer used internally by file-streaming ops.
 *  @ingroup noodle_public
 *  Must be called before conv/FCN variants that read from files.
 *  One temp buffer is needed for operations that reads from a variable.
 *  Hence, only output accumulator buffe is needed.
 *  For C*W*W tensor, the buffer should be W*W
 *  @param b2 Buffer #2 (float accumulator). See size guidance above.
 */
void noodle_setup_temp_buffers(void *b2);

/** 
 *  @name File and File-System Utilities 
 */

/** 
 *  @brief Initialize SD/FS backend (pins variant is meaningful only for SD_MMC). 
 *  @ingroup noodle_public
 */
bool noodle_fs_init(uint8_t clk_pin, uint8_t cmd_pin, uint8_t d0_pin);
/** 
 * @brief Initialize SD/FS backend with default pins/settings. 
 *  @ingroup noodle_public
 */
bool noodle_fs_init(uint8_t clk_pin, uint8_t cmd_pin, uint8_t d0_pin, uint8_t d1_pin, uint8_t d2_pin, uint8_t d3_pin);
/** 
 *  Initialize SD/FS backend with default pins/settings. 
 *  @ingroup noodle_public
 */
bool noodle_fs_init();
/** 
 *  @brief Initialize SD/FS backend with a specific CS_PIN. 
 *  @ingroup noodle_public
 */
bool noodle_fs_init(uint8_t cs_pin);

/** @brief Read the first line of a given text file.
 *  @ingroup noodle_public
 *  @param fn          File name to read.
 *  @param line        Reading result.
 *  @param maxlen      Maximum character length to read.
 */
void noodle_read_top_line(const char* fn, char *line, size_t maxlen);

/** @brief Delete a file if it exists. 
 *  @ingroup noodle_internal
 */
void noodle_delete_file(const char *fn);

/** 
 *  @brief Read bytes from a file until a terminator or length-1 (NULL terminated).
 *  @ingroup noodle_public
 *  @param file Open file handle.
 *  @param terminator Stop when this character is read (not stored).
 *  @param buffer Destination buffer (will always be NULL terminated).
 *  @param length Maximum bytes to write into @p buffer including the NULL.
 *  @return Number of characters written (excluding NULL).
 */
size_t noodle_read_bytes_until(NDL_File &file, char terminator, char *buffer, size_t length);

/** 
 *  @name Scalar I/O helpers 
 */
 
/** Write a float followed by a newline (human-readable). 
 *  @ingroup noodle_internal
 */
void noodle_write_float(NDL_File &f, float d);

/** 
 *  @brief Read a float up to the next newline. 
 *  @ingroup noodle_internal
 */
float noodle_read_float(NDL_File &f);

/** 
 *  @brief Read a byte value fron an opened file handler and store  as an integer text line. 
 *  @ingroup noodle_internal
 */
byte noodle_read_byte(NDL_File &f);

/** 
 *  @brief Write a byte value as an integer text line to an opend file. 
 *  @ingroup noodle_internal
 */
void noodle_write_byte(NDL_File &f, byte d);

/** 
 *  @name Memory utilities 
 */

/** 
 *  @brief Allocate a raw float buffer of @p size bytes. 
 *  @ingroup noodle_public
 */
float *noodle_create_buffer(uint16_t size);

/** 
 *  @brief Free a buffer allocated by noodle_create_buffer.
 *  @ingroup noodle_public
 */
void noodle_delete_buffer(float *buffer);

/** 
 *  @brief Fill @p buffer with zeros (n floats). 
 *  @ingroup noodle_internal
 */
void noodle_reset_buffer(float *buffer, uint16_t n);

/** 
 *  Slice a stacked [Z, W, W] tensor laid out as contiguous planes.
 *  @ingroup noodle_internal
 *  @param flat Pointer to base of the contiguous array.
 *  @param W    Width/height of each 2D plane.
 *  @param z    Plane index to slice.
 *  @return Pointer to the start of plane @p z (no bounds checks).
 */
float* noodle_slice(float* flat, size_t W, size_t z);

/** 
 *  @brief Write an array of @p n floats to @p fn, one value per line.
 *  File will be opened and closed.
 *  @ingroup noodle_public
 */
void noodle_array_to_file(float *array, const char *fn, uint16_t n);

/** 
 *  @brief Write an array of @p n floats to @p fo (an opened file handler), one value per line.
 *  No file open and close operations.
 *  @ingroup noodle_internal
 */
void noodle_array_to_file(float *array, NDL_File &fo, uint16_t n);

/** 
 *  @brief Write an @p n×@p n byte grid to @p fn as bytes, row-major.
 *  File will be opened and closed.
 *  @ingroup noodle_public
 */
void noodle_grid_to_file(byte *grid, const char *fn, uint16_t n);

/** 
 *  @brief Write an @p n×@p n byte grid to @p fo (opened file handler) as bytes, row-major.
 *  No file open and close operations.
 *  @ingroup noodle_internal
 */
void noodle_grid_to_file(byte *grid, NDL_File &fo, uint16_t n);

/** 
 *  @brief Write an @p n×@p n float grid to @p fn, row-major.
 *  @ingroup noodle_public
 */
void noodle_grid_to_file(float *grid, const char *fn, uint16_t n);

/** 
 *  @brief Write an @p n×@p n float grid to @p fo (an opened file handler), row-major.
 *  @ingroup noodle_internal
 */
void noodle_grid_to_file(float *grid, NDL_File &fo, uint16_t n);

/** 
 *  @brief Read a float array of length @p K from @p fn (one value per line). 
 *  @ingroup noodle_public
 */
void noodle_array_from_file(const char *fn, float *buffer, uint16_t K);

/** 
 *  @brief Read a float array of length @p K from an opened file handler @p fi (one value per line). 
 *  @ingroup noodle_public
 */
void noodle_array_from_file(NDL_File &fi, float *buffer, uint16_t K);

/** 
 * Read an @p K×@p K byte grid (stored as float) from @p fn into @p buffer.
 *  @ingroup noodle_public
 */
void noodle_grid_from_file(const char *fn, byte *buffer, uint16_t K);

/** 
 *  @brief Read an @p K × @p K grid (stored as byte) from @p fi (opened file handler) into @p buffer.
 *  @ingroup noodle_public
 */
void noodle_grid_from_file(NDL_File &fi, byte *buffer, uint16_t K);

/** 
 *  @brief Read an @p K × @p K grid (stored as float) from @p fn into @p buffer.
 *  @ingroup noodle_public
 */
void noodle_grid_from_file(const char *fn, int8_t *buffer, uint16_t K);

/** 
 *  @brief Read an @p K × @p K grid (stored as int8) from @p fi (opened file handler) into @p buffer.
 *  @ingroup noodle_public
 */
void noodle_grid_from_file(NDL_File &fi, int8_t *buffer, uint16_t K);

/** 
 *  @brief Read an @p K × @p K grid (stored as float) from @p fn into @p buffer.
 *  @ingroup noodle_public
 */
void noodle_grid_from_file(const char *fn, float *buffer, uint16_t K);

/** 
 * Read an @p K × @p K grid (stored as float) from an opened file handler @p fi into @p buffer.
 *  @ingroup noodle_internal
 */
void noodle_grid_from_file(NDL_File &fi, float *buffer, uint16_t K);

/** 
 *  @name Internal Helpers for Pooling 
 */

/** 
 *  @brief 2D pooling over a V×V map, writing results to a file (one float per line).
 *  @ingroup noodle_internal
 *  Pooling mode (MAX or MEAN) is selected via NOODLE_POOL_MODE at compile time.
 *  This layer uses valid pooling. NO PADDING IS APPLIED!
 *  @param input  Input V×V map (float).
 *  @param W      V (input width/height).
 *  @param K      Pool kernel size (M).
 *  @param S      Pool stride (T).
 *  @param fn     Output filename template (receives O where applicable).
 *  @return V_out = (V - K)/S + 1.
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);
/**  
 *  @overload using NDL_File output handle. 
 *  @ingroup noodle_internal
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, NDL_File &fo);
/**  
 *  @overload using float output buffer.
 *  @ingroup noodle_internal
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, float *output);

/** 1D MAX pooling (file output). Writes V_out values (one per line) to @p fn.
 *  @ingroup noodle_internal
 *  @param input  Input length-V vector (float).
 *  @param W      V (input length).
 *  @param K      Pool kernel size.
 *  @param S      Pool stride.
 *  @param fn     Output filename (per-O when called in loops).
 *  @return V_out = (V - K)/S + 1.
 */
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);
/**  
 *  @overload using NDL_File output handle. 
 *  @ingroup noodle_internal
 */
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S, NDL_File &fo);

/**  
 *  @name 2D Convolution
 *
 *  Packed-file conventions:
 *  - Input file: packed CHW planes (each plane W×W).
 *  - Output file: packed CHW planes (each plane V_out×V_out) in output-channel order.
 *
 *  Padding:
 *  - Noodle uses symmetric, stride-independent padding.
 *  - If P >= 0, that value is used as the symmetric padding on all sides.
 *  - If P == 65535, padding is computed automatically as:
 *      P = floor((K - 1) / 2)
 *    which preserves spatial size when S = 1 and K is odd.
 *
 *  Requires temporary buffers set via ::noodle_setup_temp_buffers.
 *  Buffer sizes are typically W×W floats, used as per-channel scratch space.
 */

/** 
 *  @brief File→File 2D conv with BYTE input feature maps.
 *  @ingroup noodle_public
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
 *  @brief File→File 2D conv with FLOAT input feature maps. 
 *  @ingroup noodle_public
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
 *  @brief File→Memory 2D conv with FLOAT inputs; writes [O, Wo, Wo] tensor to output. 
 *  @ingroup noodle_public
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
 *  @brief Memory→File 2D conv with FLOAT inputs and in-file conv parameters. 
 *  @ingroup noodle_public
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
 *  @brief Memory→File 2D conv with FLOAT inputs and in-varibale conv parameters 
 *  @ingroup noodle_public
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
 *  @brief Memory→Memory 2D conv with FLOAT inputs and in-file conv parameters. 
 *  @ingroup noodle_public
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
 * Memory→Memory 2D conv with FLOAT inputs and in-variable conv parameters. 
 *  @ingroup noodle_public
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
 *  @name 1D Convolution 
 *
 *  - Conv.K used as kernel length
 *  - W used as input length
 */
 
/**
 *  File CHW→File CHW 1D convolution with optional bias+activation and a pooling stage.
 *  @ingroup noodle_public
 *
 *  This follows the same I/O convention as noodle_conv_float():
 *  - @p in_fn is a *single* packed input file containing all input channels in CHW order
 *    (for 1D: C then W samples, one channel after another).
 *  - @p out_fn is a *single* packed output file; for each output channel O we append either the
 *    pooled sequence or the raw sequence (depending on overload).
 *  - Weights are read sequentially from @p conv.weight_fn in the order: for O in [0..n_outputs)
 *    and I in [0..n_inputs), read K floats (kernel taps).
 *  - Biases are read sequentially from @p conv.bias_fn (one float per output channel).
 *
 *  @param in_fn       Packed input filename (CHW).
 *  @param n_inputs    Number of input channels I.
 *  @param out_fn      Packed output filename (CHW).
 *  @param n_outputs   Number of output channels O.
 *  @param W           Input length.
 *  @param conv        Convolution parameters (K, P, S, weight_fn, bias_fn, act).
 *  @param pool        Pool parameters (kernel M, stride T).
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return V_out after pooling.
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
 *  File CHW→File CHW 1D convolution with bias+activation and NO pooling stage.
 *  @ingroup noodle_public
 *  Semantics as above but appends raw conv+bias(+ReLU) sequences for each output channel to
 *  @p out_fn.
 *  @param in_fn       Packed input filename (CHW).
 *  @param n_inputs    Number of input channels I.
 *  @param out_fn      Packed output filename (CHW).
 *  @param n_outputs   Number of output channels O.
 *  @param W           Input length.
 *  @param conv        Convolution parameters (K, P, S, weight_fn, bias_fn, act).
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return V (pre-pooling output length).
 */
uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const Conv &conv,
                       CBFPtr progress_cb=NULL);


/** 
 *  Memory→Memory 1D convolution with optional bias+activation and a pooling stage.
 *  @ingroup noodle_public
 *  @param in          Input array (CHW).
 *  @param n_inputs    Number of input channels I.
 *  @param out         Output array (CHW).
 *  @param n_outputs   Number of output channels O.
 *  @param W           Input length.
 *  @param conv        Convolution parameters (K, P, S, weight_fn, bias_fn, act).
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return V_out after pooling.
 */
uint16_t noodle_conv1d(const float *in,
                       uint16_t n_inputs,
                       float *out,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb=NULL);

/** 
 * @name Internal Helpers for Convolution 
 */
                       
/** 
 *  1D convolution with zero padding/stride, accumulating into @p output.
 *  Output length is `V = (W - K + 2P)/S + 1`.
 *  @ingroup noodle_internal
 */
uint16_t noodle_do_conv1d(float *input, float *kernel, uint16_t W, uint16_t K,
                          float *output, uint16_t P, uint16_t S);

/** 
 *  @brief 2D valid/same convolution with zero padding and stride, accumulating into @p output.
 *  @ingroup noodle_internal
 *  Output spatial size is `V = (W - K + 2P)/S + 1`.
 *  @param grid           Input grid (bytes interpreted as values 0..255).
 *  @param kernel         K×K float kernel.
 *  @param K,W,P,S        See common semantics.
 *  @param output  Accumulator buffer of size at least V×V (float).
 *  @return V, the output width/height.
 */
uint16_t noodle_do_conv(byte *grid, const float *kernel, uint16_t K, uint16_t W,
                        float *output, uint16_t P, uint16_t S);
/**  
 *  @overload using float input grid. 
 *  @ingroup noodle_internal
 */
uint16_t noodle_do_conv(float *grid, const float *kernel, uint16_t K, uint16_t W,
                        float *output, uint16_t P, uint16_t S);

/**
 *  Add a scalar bias to each element of a V×V map in-place and apply ReLU.
 *  @ingroup noodle_internal
 *  Convenience legacy helper used by file-streamed conv paths.
 *  @param output Accumulator buffer of size V×V (float).
 *  @param bias   Scalar to add to each element.
 *  @param n      V (output width/height).
 *  @return V.
 */
uint16_t noodle_do_bias(float *output, float bias, uint16_t n);

/** 
 *  @brief Add bias to each element of a V×V map (in-place) and optionally apply activation.
 *  @ingroup noodle_internal
 *  @param output Accumulator buffer of size V×V (float).
 *  @param bias   Scalar to add to each element.
 *  @param n      V (output width/height).
 *  @param act    Activation to apply (ACT_NONE or ACT_RELU).
 *  @return V.
 */
uint16_t noodle_do_bias_act(float *output, float bias, uint16_t n, Activation act);

/** 
 *  @brief Get padded input sample from a byte grid with zero padding.
 *  @ingroup noodle_internal
 *  Noodle uses symmetric, stride-independent padding for all convolutions.
 *  @param grid  Input @p W×@p W bytes.
 *  @param i,j   Padded coordinates in [0, W+2P).
 *  @param W,P   See common semantics.
 *  @return Grid value as float, or 0 outside bounds.
 */
float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P);

/** 
 *  @overload 
 *  @brief Get padded input sample from a float grid with zero padding.
 *  @ingroup noodle_internal
 */
float noodle_get_padded_x(float *grid, int16_t i, int16_t j, int16_t W, int16_t P);


/** 
 * @name Activations 
 */
 
/** 
 *  In-place softmax over a length-@p n vector. Returns @p n. 
 *  @ingroup noodle_public
 */
uint16_t noodle_soft_max(float *input_output, uint16_t n);
/** 
 *  In-place sigmoid over a length-@p n vector. Returns @p n.
 *  @ingroup noodle_public
 */
uint16_t noodle_sigmoid(float *input_output, uint16_t n);
/** 
 *  In-place ReLU over a length-@p n vector. Returns @p n.
 *  @ingroup noodle_public
 */
uint16_t noodle_relu(float *input_output, uint16_t n);

/** 
 *  FCN parameters (plain filenames; no tokenization). 
 *  @ingroup noodle_public
 */
struct FCN {
  const char *weight_fn = nullptr;
  const char *bias_fn   = nullptr;
  Activation act = ACT_RELU;
};

/** 
 *  FCN parameters (filenames; no tokenization). 
 *  @ingroup noodle_public
 */
struct FCNFile {
  const char *weight_fn = nullptr;
  const char *bias_fn   = nullptr;
  Activation act = ACT_RELU;
};

/** 
 *  FCN parameters for in-variable weights/bias (row-major weights [n_outputs, n_inputs]). 
 *  @ingroup noodle_public
 */
struct FCNMem {
  const float *weight = nullptr;
  const float *bias   = nullptr;
  Activation act = ACT_RELU;
};

/** 
 * @name Fully Connected Network 
 */

/** 
 *  Memory→File fully-connected layer (int8 inputs; weights/bias from files).
 *  @ingroup noodle_public
 *  Computes y = W·x + b, optionally applies ReLU, and writes @p n_outputs lines to @p out_fn.
 *  @param input       Pointer to @p n_inputs int8 values.
 *  @param n_inputs    Number of inputs.
 *  @param n_outputs   Number of outputs.
 *  @param out_fn      Output filename (one float per line).
 *  @param fcn         Filenames for weights and bias; weights read row-major [O, I].
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return n_outputs.
 */
uint16_t noodle_fcn(const int8_t *input, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);


/** 
 *  File→File fully-connected layer (float text inputs; params from files).
 *  @ingroup noodle_public
 *  For each output neuron O, rewinds @p in_fn, accumulates dot(W[O], x) + b[O],
 *  applies activation, and appends to @p out_fn.
 *  @param in_fn       Input filename containing @p n_inputs floats (one per line).
 *  @param n_inputs    Number of inputs.
 *  @param n_outputs   Number of outputs.
 *  @param out_fn      Output filename (appends/overwrites as created).
 *  @param fcn         Filenames for weights/bias.
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return n_outputs.
 */
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);


/** 
 *  Memory→Memory fully-connected layer (float inputs; explicit in-variable weights/bias).
 *  @ingroup noodle_public
 *  Weights are row-major `[n_outputs, n_inputs]` and biases length @p n_outputs.
 *  @param input       Float array of length @p n_inputs.
 *  @param n_inputs    Number of inputs.
 *  @param n_outputs   Number of outputs.
 *  @param output      Float array of length @p n_outputs (written).
 *  @param fcn         in-variable weights/bias and activation.
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return n_outputs.
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNMem &fcn,
                    CBFPtr progress_cb = NULL);


/** 
 *  Memory→Memory fully-connected layer (byte inputs; params from files).
 *  @ingroup noodle_public
 *  @param input       Byte array of length @p n_inputs (0..255 interpreted as float).
 *  @param n_inputs    Number of inputs.
 *  @param n_outputs   Number of outputs.
 *  @param output      Float array of length @p n_outputs (written).
 *  @param fcn         Filenames for weights/bias and activation mode.
 *  @param progress_cb Optional progress callback.
 *  @return n_outputs.
 */
uint16_t noodle_fcn(const byte *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);


/** 
 *  Memory→Memory fully-connected layer (int8 inputs; params from files).
 *  @ingroup noodle_public
 *  @param input       Int8 array of length @p n_inputs.
 *  @param n_inputs    Number of inputs.
 *  @param n_outputs   Number of outputs.
 *  @param output      Float array of length @p n_outputs (written).
 *  @param fcn         Filenames for weights/bias and activation mode.
 *  @param progress_cb Optional progress callback.
 *  @return n_outputs.
 */
uint16_t noodle_fcn(const int8_t *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);


/** 
 *  File→Memory fully-connected layer (float output; params from files).
 *  @ingroup noodle_public
 *  Reads inputs from @p in_fn for each output neuron O, computing y[O] = dot(W[O], x) + b[O],
 *  then applies activation.
 *  @param in_fn       Input filename with @p n_inputs floats per forward pass.
 *  @param n_inputs    Number of inputs.
 *  @param n_outputs   Number of outputs.
 *  @param output      Float array of length @p n_outputs (written).
 *  @param fcn         Filenames for weights/bias.
 *  @param progress_cb Optional progress callback.
 *  @return n_outputs.
 */
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);

/** 
 *  Memory→Memory fully-connected layer (float output; params from files).
 *  @ingroup noodle_public
 *  Reads inputs from @p input for each output neuron O, computing y[O] = dot(W[O], x) + b[O],
 *  then applies activation.
 *  @param input       Float array of length @p n_inputs.
 *  @param n_inputs    Number of inputs.
 *  @param n_outputs   Number of outputs.
 *  @param output      Float array of length @p n_outputs (written).
 *  @param fcn         Filenames for weights/bias.
 *  @param progress_cb Optional progress callback.
 *  @return n_outputs.
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb);

/** 
 *  Memory→File fully-connected layer (float inputs; params from files).
 *  @ingroup noodle_public
 *  Computes y = W·x + b, optionally applies activation, and writes @p n_outputs lines to @p out_fn.
 *  @param input       Pointer to @p n_inputs float values.
 *  @param n_inputs    Number of inputs.
 *  @param n_outputs   Number of outputs.
 *  @param out_fn      Output filename (one float per line).
 *  @param fcn         Filenames for weights and bias; weights read row-major [O, I].
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return n_outputs.
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const FCNFile &fcn,
                    CBFPtr progress_cb);

/** 
 *  @name Tensor Reshaping 
 */

/** 
 *  File→Memory flatten: reads @p n_filters feature maps from files named by @p in_fn
 *  @ingroup noodle_public
 *  (tokenized by O via ::noodle_n2ll at positions 4/6 as appropriate) and writes a vector
 *  of length V×V×n_filters in row-major [i* n_filters + k].
 *  @param in_fn      Base filename of pooled feature maps (receives O).
 *  @param output     Output buffer of length V×V×n_filters.
 *  @param V          Spatial size (width=height).
 *  @param n_filters  Number of channels (O).
 *  @return V×V×n_filters.
 */
uint16_t noodle_flat(const char *in_fn, float *output, uint16_t V, uint16_t n_filters);

/** 
 *  Memory→Memory flatten: flattens [O, V, V] into a vector of length V×V×n_filters.
 *  @ingroup noodle_public
 *  @param input      Base pointer to stacked feature maps [O, V, V].
 *  @param output     Output buffer of length V×V×n_filters.
 *  @param V          Spatial size.
 *  @param n_filters  Number of channels O.
 *  @return V×V×n_filters.
 */
uint16_t noodle_flat(float *input, float *output, uint16_t V, uint16_t n_filters);

/**
 *  @ingroup noodle_public
 *  Global Average Pooling for a channel-first tensor in memory.
 *  @param x_chw Pointer to the input tensor in [C][W][W] layout.
 *  @param C Number of channels.
 *  @param W Width/height of each channel plane.
 *  @return Number of channels C
 */
uint16_t noodle_gap(float *inout,
                    uint16_t C,
                    uint16_t W);

/**
 *  @ingroup noodle_public
 *  Find the maximum value and its index in a float array.
 *  @param input Pointer to the input float array.
 *  @param n Length of the input array.
 *  @param max_val Reference to store the maximum value found.
 *  @param max_idx Reference to store the index of the maximum value.
 */
void noodle_find_max(float *input,
                     uint16_t n,
                     float &max_val,
                     uint16_t &max_idx);

/** 
 *  @name 2D Depth-wise Convolution 
 */

/** 
 *  Depthwise convolution (float input/output; params from files).
 *  @ingroup noodle_public
 *  For each input channel I, reads the I-th input feature map from @p in_fn (tokenized by I),
 *  convolves it with the depthwise kernel read from @p conv.weight_fn (also tokenized by I),
 *  adds bias from @p conv.bias_fn (one bias per input channel), applies activation,
 *  and writes the output feature map to @p out_fn (tokenized by I).
 *  Requires temp buffers set via ::noodle_setup_temp_buffers.
 *  @param in_fn       Base input filename template (receives I).
 *  @param n_channels  Number of input/output channels.
 *  @param out_fn      Base output filename template (receives I).
 *  @param W           Input width/height.
 *  @param conv        Convolution parameters (K, P, S, weight_fn, bias_fn, act).
 *  @param pool        Pool parameters (kernel M, stride T).
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return Output width after pooling.
 */
uint16_t noodle_dwconv_float(const char *in_fn,
                             uint16_t n_channels,
                             const char *out_fn,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb);


/** 
 *  Memory → memory depthwise conv (float input).
 *  @ingroup noodle_public
 *  Assumes:
 *  - input layout:  [C][W][W] flattened
 *  - output layout: [C][Wo][Wo] flattened (Wo depends on pooling)
 *  @param input       Pointer to the input tensor in [C][W][W] layout.
 *  @param n_channels  Number of input/output channels.
 *  @param output      Pointer to the output tensor in [C][Wo][Wo] layout.
 *  @param W           Input width/height.
 *  @param conv        Convolution parameters (K, P, S, weight_fn, bias_fn, act).
 *  @param pool        Pool parameters (kernel M, stride T).
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return Output width after pooling.  
 */
uint16_t noodle_dwconv_float(float *input,
                             uint16_t n_channels,
                             float *output,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb);
/**
 *  @ingroup noodle_public
 *  Memory → memory depthwise conv (float input) with in-variable weights/bias. 
 *  Assumes:
 *   - input layout:  [C][W][W] flattened
 *   - output layout: [C][Wo][Wo] flattened (Wo depends on pooling )
 *  @param input       Pointer to the input tensor in [C][W][W] layout.
 *  @param n_channels  Number of input/output channels.
 *  @param output      Pointer to the output tensor in [C][Wo][Wo] layout.
 *  @param W           Input width/height.
 *  @param conv        Convolution parameters with in-variable weights/bias.
 *  @param pool        Pool parameters (kernel M, stride T).
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return Output width after pooling.  
 */                             
uint16_t noodle_dwconv_float(float *input,
                             uint16_t n_channels,
                             float *output,
                             uint16_t W,
                             const ConvMem &conv,
                             const Pool &pool,
                             CBFPtr progress_cb);                            

/**
 *  @ingroup noodle_internal
 *  Unpack batch normalization parameters from a flat array.
 *  @param bn_params Pointer to the packed batch normalization parameters.
 *  @param C Number of channels.
 *  @param gamma Output pointer to the per-channel scale parameters.
 *  @param beta Output pointer to the per-channel shift parameters.
 *  @param mean Output pointer to the per-channel mean parameters.
 *  @param var Output pointer to the per-channel variance parameters.
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
 *  @ingroup noodle_public
 *  Batch Normalization for a channel-first tensor in memory.
 *  @param x Pointer to the input tensor in [C][W][W] layout.
 *  @param C Number of channels.
 *  @param W Width/height of each channel plane.
 *  @param gamma Pointer to the per-channel scale parameters.
 *  @param beta Pointer to the per-channel shift parameters.
 *  @param mean Pointer to the per-channel mean parameters.
 *  @param var Pointer to the per-channel variance parameters.
 *  @param eps Small constant to avoid division by zero. 
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
 *  Batch Normalization followed by ReLU for a channel-first tensor in memory.
 *  @param x Pointer to the input tensor in [C][W][W] layout.
 *  @param C Number of channels.
 *  @param W Width/height of each channel plane.
 *  @param gamma Pointer to the per-channel scale parameters.
 *  @param beta Pointer to the per-channel shift parameters.
 *  @param mean Pointer to the per-channel mean parameters.
 *  @param var Pointer to the per-channel variance parameters.
 *  @param eps Small constant to avoid division by zero. 
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
 *  @ingroup noodle_public
 *  Batch Normalization for a channel-first tensor in memory.
 *  @param x Pointer to the input tensor in [C][W][W] layout.
 *  @param C Number of channels.
 *  @param W Width/height of each channel plane.
 *  @param bn_params Pointer to the packed batch normalization parameters.
 *  @param eps Small constant to avoid division by zero.
 */ 
uint16_t noodle_bn(float *x,
                   uint16_t C,
                   uint16_t W,
                   const float *bn_params,
                   float eps=1e-3);

/*
 *  Batch Normalization followed by ReLU for a channel-first tensor in memory.
 *  @param x Pointer to the input tensor in [C][W][W] layout.
 *  @param C Number of channels.
 *  @param W Width/height of each channel plane.
 *  @param bn_params Pointer to the packed batch normalization parameters.
 *  @param eps Small constant to avoid division by zero.
 */
uint16_t noodle_bn_relu(float *x,
                        uint16_t C,
                        uint16_t W,
                        const float *bn_params,
                        float eps=1e-3);

