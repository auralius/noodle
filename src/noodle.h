/**
 * @file noodle.h
 * @brief File-streamed CNN/ML primitives for tiny MCUs with pluggable filesystem backends.
 *
 * @details
 * Noodle provides minimal convolution, pooling, flatten, fully-connected and activation
 * routines that operate either entirely in memory or by streaming tensors/parameters to
 * and from a filesystem (SD/FFat/SD_MMC). Backends are selected in @ref noodle_fs.h via
 * preprocessor flags. The library is designed for extremely small RAM budgets, so most
 * file-based APIs reuse two caller-supplied temporary buffers 
 * (see @ref noodle_setup_temp_buffers).
 *
 * ### Backends
 * Select exactly one of `NOODLE_USE_SDFAT`, `NOODLE_USE_SD_MMC`, or `NOODLE_USE_FFAT`, 
 * or `NOODLE_USE_LITTLEFS`, or `NOODLE_USE_LITTLEFS`, or `NOODLE_USE_LITTLEFS` before
 * including this header; see @ref noodle_fs.h for details.
 *
 * ### File naming convention (2-letter indices)
 * Some routines iterate input (I) and output (O) channel files by writing a two-letter base-26
 * code into filename templates using ::noodle_n2ll. By convention the 5th and 7th bytes
 * (0‑based indexes 4 and 6) of `in_fn`, `weight_fn`, and `out_fn` are overwritten with the
 * two-letter tokens for **I** and **O**, respectively. Ensure your templates have writable
 * space there (e.g., `"in____"`, `"w____"`, `"out__"`). See function docs.
 *
 * ### Temporary buffers
 * Call ::noodle_setup_temp_buffers() once prior to convolution/FCN routines that read from files.
 * Unless otherwise stated:
 *  - temp buffer #1: at least `W*W*sizeof(input_element)` bytes
 *  - temp buffer #2: at least `W*W*sizeof(float)` bytes (accumulator)
 *
 * ### Pooling mode
 * 2D pooling mode is compile-time selectable via @ref NOODLE_POOL_MODE in @ref noodle_config.h
 * (MAX or MEAN).
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

/** 2D convolution parameters. */
enum Activation : uint8_t { ACT_NONE = 0, ACT_RELU = 1 , ACT_SOFTMAX = 2};

struct Conv {
  // Shared conv params for 1D/2D
  uint16_t K;        ///< Kernel size
  uint16_t P = 0;    ///< Padding
  uint16_t S = 1;    ///< Stride

  // File-backed weights/bias filenames
  const char *weight_fn = nullptr;
  const char *bias_fn   = nullptr;

  // Activation to apply after bias
  Activation act = ACT_RELU;
};


/** 2D pooling parameters. Use enabled=false for identity (no pooling). */
struct Pool {
  uint16_t M = 2;    ///< Pool kernel
  uint16_t T = 2;    ///< Pool stride
};



/** Progress callback type used by long-running routines.
 *  @param progress A normalized progress in [0,1], monotonically nondecreasing.
 */
typedef void (*CBFPtr)(float progress);

/** @name Common parameter semantics
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
///@{
///@}

// ============================================================================
// Utilities
// ============================================================================

/** Slice a stacked [Z, W, W] tensor laid out as contiguous planes.
 *  @param flat Pointer to base of the contiguous array.
 *  @param W    Width/height of each 2D plane.
 *  @param z    Plane index to slice.
 *  @return Pointer to the start of plane @p z (no bounds checks).
 */
inline float* noodle_slice(float* flat, size_t W, size_t z);

/** Provide two reusable temporary buffers used internally by file-streaming ops.
 *  Must be called before conv/FCN variants that read from files.
 *  @param b1 Buffer #1 (input scratch). See size guidance above.
 *  @param b2 Buffer #2 (float accumulator). See size guidance above.
 */
void   noodle_setup_temp_buffers(void *b1, void *b2 = NULL);

// ============================================================================
// Filesystem helpers
// ============================================================================

/** Open a file for write (removing an existing file first). */
NDL_File noodle_open_file_for_write(const char* fn);

/** Read bytes from a file until a terminator or length-1, NUL-terminating the buffer.
 *  @param file Open file handle.
 *  @param terminator Stop when this character is read (not stored).
 *  @param buffer Destination buffer (will always be NUL terminated).
 *  @param length Maximum bytes to write into @p buffer including the NUL.
 *  @return Number of characters written (excluding NUL).
 */
size_t   noodle_read_bytes_until(NDL_File &file, char terminator, char *buffer, size_t length);

/** Initialize SD/FS backend (pins variant is meaningful only for SD_MMC). */
bool noodle_sd_init(int clk_pin, int cmd_pin, int d0_pin);
/** Initialize SD/FS backend with default pins/settings. */
bool noodle_sd_init();
/** Initialize SD/FS backend with a specific CS_PIN. */
bool noodle_sd_init(uint8_t cs_pin);

/** Convert an integer 0..675 into a two-letter base-26 code ("aa".."zz").
 *  @param number Index to encode.
 *  @param out    Destination buffer of at least 2 bytes (no terminator written).
 */
void noodle_n2ll(uint16_t number, char *out);

// ============================================================================
// Scalar I/O helpers
// ============================================================================

/** Write a float followed by a newline (human-readable). */
void    noodle_write_float(NDL_File &f, float d);
/** Read a float up to the next newline. */
float   noodle_read_float(NDL_File &f);
/** Read a byte value stored as an integer text line. */
byte    noodle_read_byte(NDL_File &f);
/** Write a byte value as an integer text line. */
void    noodle_write_byte(NDL_File &f, byte d);
/** Delete a file if it exists. */
void    noodle_delete_file(const char *fn);

// ============================================================================
// Memory utilities
// ============================================================================

/** Allocate a raw float buffer of @p size bytes. */
float *noodle_create_buffer(uint16_t size);
/** Free a buffer allocated by ::noodle_create_buffer. */
void   noodle_delete_buffer(float *buffer);
/** Fill @p buffer with zeros (n floats). */
void   noodle_reset_buffer(float *buffer, uint16_t n);

// ============================================================================
// Array/Grid I/O
// ============================================================================

/** Write an array of @p n floats to @p fn, one value per line. */
void noodle_array_to_file(float *array, const char *fn, uint16_t n);
void noodle_array_to_file(float *array, NDL_File &fo, uint16_t n);
/** Write an @p n×@p n byte grid to @p fn as floats, row-major. */
void noodle_grid_to_file(byte *grid, const char *fn, uint16_t n);
void noodle_grid_to_file(byte *grid, NDL_File &fo, uint16_t n);
/** Write an @p n×@p n float grid to @p fn, row-major. */
void noodle_grid_to_file(float *grid, const char *fn, uint16_t n);
void noodle_grid_to_file(float *grid, NDL_File &fo, uint16_t n);

/** Read a float array of length @p K from @p fn (one value per line). */
void noodle_array_from_file(const char *fn, float *buffer, uint16_t K);
void noodle_array_from_file(NDL_File &fi, float *buffer, uint16_t K);
/** Read an @p K×@p K byte grid (stored as floats) from @p fn into @p buffer. */
void noodle_grid_from_file(const char *fn, byte *buffer, uint16_t K);
void noodle_grid_from_file(NDL_File &fi, byte *buffer, uint16_t K);
/** Read an @p K×@p K int8_t grid (stored as floats) from @p fn into @p buffer. */
void noodle_grid_from_file(const char *fn, int8_t *buffer, uint16_t K);
void noodle_grid_from_file(NDL_File &fi, int8_t *buffer, uint16_t K);
/** Read an @p K×@p K float grid from @p fn into @p buffer. */
void noodle_grid_from_file(const char *fn, float *buffer, uint16_t K);
void noodle_grid_from_file(NDL_File &fi, float *buffer, uint16_t K);

/** Return padded input sample from a byte grid with zero padding.
 *  @param grid  Input @p W×@p W bytes.
 *  @param i,j   Padded coordinates in [0, W+2P).
 *  @param W,P   See common semantics.
 *  @return Grid value as float, or 0 outside bounds.
 */
float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P);
/** @overload */
float noodle_get_padded_x(float *grid, int16_t i, int16_t j, int16_t W, int16_t P);

// ============================================================================
// 2D Convolution (primitive + streamed variants)
// ============================================================================

/** 2D valid/same convolution with zero padding and stride, accumulating into @p output.
 *  Output spatial size is `V = (W - K + 2P)/S + 1`.
 *  @param grid           Input grid (bytes interpreted as values 0..255).
 *  @param kernel         K×K float kernel.
 *  @param K,W,P,S        See common semantics.
 *  @param output  Accumulator buffer of size at least V×V (float).
 *  @return V, the output width/height.
 */
uint16_t noodle_do_conv(byte *grid, float *kernel, uint16_t K, uint16_t W,
                        float *output, uint16_t P, uint16_t S);
/** @overload using float input grid. */
uint16_t noodle_do_conv(float *grid, float *kernel, uint16_t K, uint16_t W,
                        float *output, uint16_t P, uint16_t S);

/** File→File 2D conv with byte input feature maps.
 *  For each output channel O: accumulates over all input channels I using `weight_fn`
 *  (which receives both I and O two-letter indices), then adds bias from `bias_fn`,
 *  applies ReLU (if bias+ReLU stage is included in later pooling step), and writes pooled
 *  feature map to `out_fn` (receives O index). Progress callback is optional.
 *  Requires temp buffers set via ::noodle_setup_temp_buffers.
 *  @return Output width after pooling.
 */

// --- Internal helpers (exposed for embedded builds) ---

/** Add a scalar bias to each element of a V×V map in-place and apply ReLU.
 *  Convenience legacy helper used by file-streamed conv paths.
 *  @param output Accumulator buffer of size V×V (float).
 *  @param bias   Scalar to add to each element.
 *  @param n      V (output width/height).
 *  @return V.
 */
uint16_t noodle_do_bias(float *output, float bias, uint16_t n);

/** Add a scalar bias to each element of a V×V map in-place and optionally apply activation.
 *  @param output Accumulator buffer of size V×V (float).
 *  @param bias   Scalar to add to each element.
 *  @param n      V (output width/height).
 *  @param act    Activation to apply (ACT_NONE or ACT_RELU).
 *  @return V.
 */
uint16_t noodle_do_bias_act(float *output, float bias, uint16_t n, Activation act);

/** 2D pooling over a V×V map, writing results to a file (one float per line).
 *  Pooling mode (MAX or MEAN) is selected via NOODLE_POOL_MODE at compile time.
 *  @param input  Input V×V map (float).
 *  @param W      V (input width/height).
 *  @param K      Pool kernel size (M).
 *  @param S      Pool stride (T).
 *  @param fn     Output filename template (receives O where applicable).
 *  @return V_out = (V - K)/S + 1.
 */
uint16_t noodle_do_pooling(float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);

/** 2D pooling over a V×V map, writing results to a provided output buffer.
 *  Pooling mode (MAX or MEAN) is selected via NOODLE_POOL_MODE at compile time.
 *  @param input   Input V×V map (float).
 *  @param W       V (input width/height).
 *  @param K       Pool kernel size (M).
 *  @param S       Pool stride (T).
 *  @param output  Output buffer of size V_out×V_out.
 *  @return V_out = (V - K)/S + 1.
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, float *output);

/** 1D MAX pooling (file output). Writes V_out values (one per line) to @p fn.
 *  @param input  Input length-V vector (float).
 *  @param W      V (input length).
 *  @param K      Pool kernel size.
 *  @param S      Pool stride.
 *  @param fn     Output filename (per-O when called in loops).
 *  @return V_out = (V - K)/S + 1.
 */
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);
// --- 2D Convolution (Conv + optional Pool) ---

/** File→File 2D conv with BYTE input feature maps. */
uint16_t noodle_conv_byte(const char *in_fn,
                          uint16_t n_inputs,
                          uint16_t n_outputs,
                          const char *out_fn,
                          uint16_t W,
                          const Conv &conv,
                          const Pool &pool,
                          CBFPtr progress_cb = NULL);

/** File→File 2D conv with FLOAT input feature maps. */
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/** File→Memory 2D conv with FLOAT inputs; writes [O, Wo, Wo] tensor to output. */
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/** Memory→File 2D conv with FLOAT inputs. */
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

/** Memory→Memory 2D conv with FLOAT inputs. */
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);

// --- 1D Convolution (Conv.K used as kernel length) ---


/** File→File 1D convolution with optional bias+activation and a 1D MAX pooling stage.
 *  For each output channel O: accumulates conv over all inputs I, adds bias from @p conv.bias_fn,
 *  applies @p conv.act (ACT_NONE/ACT_RELU), and writes pooled outputs for O to @p out_fn (tokenized
 *  by O). Inputs and weights are read from files whose names are tokenized with two-letter indices
 *  via ::noodle_n2ll at positions 4 (I) and 6 (O).
 *  @param input       Scratch buffer (length ≥ W) used internally for streaming reads.
 *  @param output      Scratch/output buffer (length ≥ W) used during accumulation.
 *  @param n_inputs    Number of input channels I.
 *  @param n_outputs   Number of output channels O.
 *  @param in_fn       Base input filename template (receives I).
 *  @param out_fn      Base output filename template (receives O).
 *  @param W           Input length.
 *  @param conv        Convolution parameters (K, P, S, weight_fn, bias_fn, act).
 *  @param pool        Pool parameters (kernel M, stride T).
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return V_out after pooling.
 */
uint16_t noodle_conv1d(float *input, float *output,
                       uint16_t n_inputs, uint16_t n_outputs,
                       const char *in_fn, const char *out_fn,
                       uint16_t W, const Conv &conv, const Pool &pool,
                       CBFPtr progress_cb = NULL);


/** File→File 1D convolution with bias+activation and NO pooling.
 *  Semantics as above but writes raw conv+bias(+ReLU) per-O to @p out_fn.
 *  @param input       Scratch buffer (length ≥ W).
 *  @param output      Accumulator buffer (length ≥ W).
 *  @param n_inputs    Number of input channels I.
 *  @param n_outputs   Number of output channels O.
 *  @param in_fn       Base input filename template (receives I).
 *  @param out_fn      Base output filename template (receives O).
 *  @param W           Input length.
 *  @param conv        Convolution parameters (K, P, S, weight_fn, bias_fn, act).
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return V (pre-pooling output length).
 */
uint16_t noodle_conv1d(float *input, float *output,
                       uint16_t n_inputs, uint16_t n_outputs,
                       const char *in_fn, const char *out_fn,
                       uint16_t W, const Conv &conv,
                       CBFPtr progress_cb = NULL);

/** File→File FCN with float inputs. */
/** File→Memory FCN with byte inputs. */
/** Memory→Memory FCN with byte inputs. */
/** Memory→Memory FCN with float inputs. */
/** File→Memory FCN; reads inputs repeatedly from @p in_fn for each output. */
/** Memory→Memory FCN with int8 inputs. */
/** File→File FCN; writes outputs to @p out_fn. */
/** Memory→Memory FCN with explicit in-memory weight & bias arrays (row-major weights).
 *  Weight layout is `[n_outputs, n_inputs]` flattened row-major.
 */
// ============================================================================
// Activations
// ============================================================================

/** In-place softmax over a length-@p n vector. Returns @p n. */
uint16_t noodle_soft_max(float *input_output, uint16_t n);
/** In-place sigmoid over a length-@p n vector. Returns @p n. */
uint16_t noodle_sigmoid(float *input_output, uint16_t n);

// ============================================================================
// 1D Convolution
// ============================================================================

/** 1D convolution with zero padding/stride, accumulating into @p output.
 *  Output length is `V = (W - K + 2P)/S + 1`.
 */
uint16_t noodle_do_conv1d(float *input, float *kernel, uint16_t W, uint16_t K,
                          float *output, uint16_t P, uint16_t S);

/** File→File 1D conv with optional 1D MAX pooling stage, writing pooled outputs per-O to files. */
/** File→File 1D conv without pooling (writes raw conv+bias(+ReLU) to files). */


/** FCN parameters (plain filenames; no tokenization). */
struct FCN {
  const char *weight_fn = nullptr;
  const char *bias_fn   = nullptr;
  Activation act = ACT_RELU;
};

/** FCN parameters (filenames; no tokenization). */
struct FCNFile {
  const char *weight_fn = nullptr;
  const char *bias_fn   = nullptr;
  Activation act = ACT_RELU;
};

/** FCN parameters for in-memory weights/bias (row-major weights [n_outputs, n_inputs]). */
struct FCNMem {
  const float *weight = nullptr;
  const float *bias   = nullptr;
  Activation act = ACT_RELU;
};

// --- Fully Connected (FCN: file-backed or in-memory) ---


/** Memory→File fully-connected layer (int8 inputs; weights/bias from files).
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


/** File→File fully-connected layer (float text inputs; params from files).
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


/** Memory→Memory fully-connected layer (float inputs; explicit in-memory weights/bias).
 *  Weights are row-major `[n_outputs, n_inputs]` and biases length @p n_outputs.
 *  @param input       Float array of length @p n_inputs.
 *  @param n_inputs    Number of inputs.
 *  @param n_outputs   Number of outputs.
 *  @param output      Float array of length @p n_outputs (written).
 *  @param fcn         In-memory weights/bias and activation.
 *  @param progress_cb Optional progress callback in [0,1].
 *  @return n_outputs.
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs, uint16_t n_outputs, float *output, const FCNMem &fcn, CBFPtr progress_cb = NULL);


/** Memory→Memory fully-connected layer (byte inputs; params from files).
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


/** Memory→Memory fully-connected layer (int8 inputs; params from files).
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


/** File→Memory fully-connected layer (float output; params from files).
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

/** Memory→Memory fully-connected layer (float output; params from files).
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
uint16_t noodle_fcn(const float *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb); 

/** File→Memory flatten: reads @p n_filters feature maps from files named by @p in_fn
 *  (tokenized by O via ::noodle_n2ll at positions 4/6 as appropriate) and writes a vector
 *  of length V×V×n_filters in row-major [i* n_filters + k].
 *  @param in_fn      Base filename of pooled feature maps (receives O).
 *  @param output     Output buffer of length V×V×n_filters.
 *  @param V          Spatial size (width=height).
 *  @param n_filters  Number of channels (O).
 *  @return V×V×n_filters.
 */
uint16_t noodle_flat(const char *in_fn, float *output, uint16_t V, uint16_t n_filters);

/** Memory→Memory flatten: flattens [O, V, V] into a vector of length V×V×n_filters.
 *  @param input      Base pointer to stacked feature maps [O, V, V].
 *  @param output     Output buffer of length V×V×n_filters.
 *  @param V          Spatial size.
 *  @param n_filters  Number of channels O.
 *  @return V×V×n_filters.
 */
uint16_t noodle_flat(float *input, float *output, uint16_t V, uint16_t n_filters);

/** Read the first line of a given text file.
 * @param fn          File name to read.
 * @param line        Reading result.
 * @param maxlen      Maximum character length to read.
 */
void noodle_read_top_line(const char* fn, char *line, size_t maxlen);

/** Depthwise convolution (float input/output; params from files).
 * For each input channel I, reads the I-th input feature map from @p in_fn (tokenized by I),
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


/** Memory → memory depthwise conv (float input).
 *  Assumes:
 *  - input layout:  [C][W][W] flattened
 *  - output layout: [C][Wo][Wo] flattened (Wo depends on pooling)
 *  - weights on FS: conv.weight_fn pattern "/w##xxyy.txt"
 *    for depthwise, we set xx == yy == channel index (so it matches your exporter decision)
 *  - bias on FS: conv.bias_fn pattern "/b##.txt" with one bias per channel
 */
uint16_t noodle_dwconv_float(float *input,
                            uint16_t n_channels,
                            float *output,
                            uint16_t W,
                            const Conv &conv,
                            const Pool &pool,
                            CBFPtr progress_cb);

/** 
 * Global Average Pooling for a channel-first tensor in memory.
 * @param x_chw Pointer to the input tensor in [C][W][W] layout.
 * @param C Number of channels.
 * @param W Width/height of each channel plane.
 */ 
void noodle_gap(float *x_chw,
                uint16_t C,
                uint16_t W);