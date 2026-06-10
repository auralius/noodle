/**
 * @file noodle_internal.h
 * @brief Private declarations shared by Noodle implementation files.
 * @ingroup noodle_internal
 *
 * Application code should include noodle.h. This header exists so the split
 * implementation files can share global file handles, scratch-buffer helpers,
 * low-level math kernels, and raw-pointer layer implementations behind the
 * public NoodleBuffer API.
 */

/**
 * @defgroup noodle_internal Internal Helpers
 * @ingroup noodle_api
 * Maintainer-facing helpers used by Noodle implementation files.
 */

#pragma once

#include "noodle.h"
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(NOODLE_USE_SDFAT)
/** @brief SdFat filesystem object used by the SdFat backend. */
extern SdFat NOODLE_FS;
#endif

/** @brief Shared weight file handle used by streaming layer implementations. */
extern NDL_File fw;
/** @brief Shared bias file handle used by streaming layer implementations. */
extern NDL_File fb;
/** @brief Shared output file handle used by streaming layer implementations. */
extern NDL_File fo;
/** @brief Shared input file handle used by streaming layer implementations. */
extern NDL_File fi;

/** @brief Global input scratch buffer, either Noodle-owned or caller-owned. */
extern void *temp_buff1;
/** @brief Global accumulation scratch buffer, either Noodle-owned or caller-owned. */
extern void *temp_buff2;

/**
 * @brief Ensure temp buffer 1 can hold a number of floats.
 * @ingroup noodle_internal
 *
 * Automatically allocated buffers grow when needed. A buffer installed with
 * noodle_setup_temp_buffers() has unknown capacity and is returned as-is.
 *
 * @param required_floats Required capacity in float elements.
 * @return Usable float pointer, or NULL on allocation failure/zero request.
 */
float *noodle_temp1_require(size_t required_floats);

/**
 * @brief Ensure temp buffer 2 can hold a number of floats.
 * @ingroup noodle_internal
 *
 * Automatically allocated buffers grow when needed. A buffer installed with
 * noodle_setup_temp_buffers() has unknown capacity and is returned as-is.
 *
 * @param required_floats Required capacity in float elements.
 * @return Usable float pointer, or NULL on allocation failure/zero request.
 */
float *noodle_temp2_require(size_t required_floats);

/**
 * @brief Free Noodle-owned scratch buffers and detach external scratch buffers.
 * @ingroup noodle_internal
 */
void noodle_temp_buffers_free(void);

/**
 * @brief Return a channel plane from a packed `[Z][W][W]` tensor.
 * @ingroup noodle_internal
 * @param flat Base pointer to the packed tensor.
 * @param W Width and height of each plane.
 * @param z Plane/channel index.
 * @return Pointer to the first element of plane @p z.
 */
float *noodle_slice(float *flat, size_t W, size_t z);

/**
 * @brief Read raw bytes from a backend file handle.
 * @ingroup noodle_internal
 * @param f Open file handle.
 * @param dst Destination buffer.
 * @param n Number of bytes to read.
 * @return Number of bytes read, or 0 for the no-filesystem backend.
 */
size_t noodle_read_raw(NDL_File &f, void *dst, size_t n);

/**
 * @brief Write raw bytes to a backend file handle.
 * @ingroup noodle_internal
 * @param f Open file handle.
 * @param src Source buffer.
 * @param n Number of bytes to write.
 * @return Number of bytes written, or 0 for the no-filesystem backend.
 */
size_t noodle_write_raw(NDL_File &f, const void *src, size_t n);

/**
 * @brief Read a block of floats using the configured scalar file format.
 * @ingroup noodle_internal
 *
 * In binary mode this reads raw float32 bytes. In text mode it calls
 * noodle_read_float() once per value.
 *
 * @param f Open input file.
 * @param dst Destination float buffer.
 * @param n_floats Number of floats requested.
 * @return Number of floats read.
 */
size_t noodle_read_float_block(NDL_File &f, float *dst, size_t n_floats);

/**
 * @brief Compute a dot product with a small unrolled loop.
 * @ingroup noodle_internal
 * @param x Input vector.
 * @param w Weight vector.
 * @param n Number of elements.
 * @return Sum of element-wise products.
 */
float noodle_dot_float_block(const float *x, const float *w, uint16_t n);

// ============================================================
// Private convolution/math helpers
// ============================================================

/**
 * @brief Accumulate one 1D convolution into an output sequence.
 * @ingroup noodle_internal
 *
 * Values outside the input sequence are treated as zero.
 *
 * @param input Input sequence with @p W values.
 * @param kernel Kernel with @p K values.
 * @param W Input sequence length.
 * @param K Kernel length.
 * @param output Accumulator receiving `V` values.
 * @param P Zero padding per side.
 * @param S Stride.
 * @return Output length before pooling.
 */
uint16_t noodle_do_conv1d(float *input, float *kernel, uint16_t W, uint16_t K,
                          float *output, uint16_t P, uint16_t S);

/**
 * @brief Apply valid 1D max pooling and write to a file.
 * @ingroup noodle_internal
 * @param input Input sequence with @p W values.
 * @param W Input sequence length.
 * @param K Pool window size.
 * @param S Pool stride.
 * @param fn Output file.
 * @return Output sequence length.
 */
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S,
                             const char *fn);

/**
 * @brief Apply valid 1D max pooling and write to an open file.
 * @ingroup noodle_internal
 * @param input Input sequence with @p W values.
 * @param W Input sequence length.
 * @param K Pool window size.
 * @param S Pool stride.
 * @param fo Open output file handle.
 * @return Output sequence length.
 */
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S,
                             NDL_File &fo);

/**
 * @brief Apply valid 1D pooling and write to memory.
 * @ingroup noodle_internal
 *
 * `K <= 1` copies the input unchanged. `S == 0` defaults the stride to @p K.
 * The helper computes mean pooling only when `NOODLE_POOL_MODE` selects mean
 * pooling; otherwise it computes max pooling.
 *
 * @param input Input sequence with @p W values.
 * @param W Input sequence length.
 * @param K Pool window size.
 * @param S Pool stride, or 0 to use @p K.
 * @param output Destination sequence.
 * @return Output sequence length.
 */
uint16_t noodle_do_pooling1d(const float *input, uint16_t W, uint16_t K,
                             uint16_t S, float *output);

/**
 * @brief Read a byte grid sample with asymmetric zero padding.
 * @ingroup noodle_internal
 * @param grid Input plane with `W * W` byte values.
 * @param i Padded-row coordinate.
 * @param j Padded-column coordinate.
 * @param W Input width and height.
 * @param P0 Top/left padding.
 * @param P1 Bottom/right padding.
 * @return Input value, or 0.0 outside the unpadded grid.
 */
float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W,
                          int16_t P0, int16_t P1);

/**
 * @brief Read a float grid sample with asymmetric zero padding.
 * @ingroup noodle_internal
 * @param grid Input plane with `W * W` float values.
 * @param i Padded-row coordinate.
 * @param j Padded-column coordinate.
 * @param W Input width and height.
 * @param P0 Top/left padding.
 * @param P1 Bottom/right padding.
 * @return Input value, or 0.0 outside the unpadded grid.
 */
float noodle_get_padded_x(float *grid, int16_t i, int16_t j, int16_t W,
                          int16_t P0, int16_t P1);

/**
 * @brief Add bias to a square output map and apply ReLU.
 * @ingroup noodle_internal
 * @param output Output map with `n * n` values.
 * @param bias Bias scalar.
 * @param n Map width and height.
 * @return @p n.
 */
uint16_t noodle_do_bias(float *output, float bias, uint16_t n);

/**
 * @brief Apply 2D pooling and write to a file.
 * @ingroup noodle_internal
 *
 * When `NOODLE_POOL_MODE` is `NOODLE_POOL_NONE`, this writes the input map
 * unchanged.
 *
 * @param input Input map with `W * W` values.
 * @param W Input width and height.
 * @param K Pool window size.
 * @param S Pool stride.
 * @param fn Output file.
 * @return Output width, or 0 for invalid pooling parameters.
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K,
                           uint16_t S, const char *fn);

/**
 * @brief Apply 2D pooling and write to an open file.
 * @ingroup noodle_internal
 * @param input Input map with `W * W` values.
 * @param W Input width and height.
 * @param K Pool window size.
 * @param S Pool stride.
 * @param fo Open output file handle.
 * @return Output width, or 0 for invalid pooling parameters.
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K,
                           uint16_t S, NDL_File &fo);

/**
 * @brief Apply 2D pooling and write to memory.
 * @ingroup noodle_internal
 * @param input Input map with `W * W` values.
 * @param W Input width and height.
 * @param K Pool window size.
 * @param S Pool stride.
 * @param output Destination map.
 * @return Output width, or @p W for identity/no pooling.
 */
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K,
                           uint16_t S, float *output);

/**
 * @brief Accumulate one byte-input 2D convolution plane.
 * @ingroup noodle_internal
 *
 * The input plane is `[W][W]`; the kernel is `[K][K]`; output is accumulated in
 * `[V][V]` order instead of cleared.
 *
 * @param grid Input plane.
 * @param kernel Kernel values.
 * @param K Kernel width.
 * @param W Input width and height.
 * @param output Output accumulator.
 * @param P Padding per side, or `65535` for SAME-style padding.
 * @param S Stride.
 * @return Output width before pooling.
 */
uint16_t noodle_do_conv(byte *grid, const float *kernel, uint16_t K,
                        uint16_t W, float *output, uint16_t P, uint16_t S);

/**
 * @brief Accumulate one float-input 2D convolution plane.
 * @ingroup noodle_internal
 *
 * The input plane is `[W][W]`; the kernel is `[K][K]`; output is accumulated in
 * `[V][V]` order instead of cleared.
 *
 * @param grid Input plane.
 * @param kernel Kernel values.
 * @param K Kernel width.
 * @param W Input width and height.
 * @param output Output accumulator.
 * @param P Padding per side, or `65535` for SAME-style padding.
 * @param S Stride.
 * @return Output width before pooling.
 */
uint16_t noodle_do_conv(float *grid, const float *kernel, uint16_t K,
                        uint16_t W, float *output, uint16_t P, uint16_t S);

/**
 * @brief Clear a float buffer.
 * @ingroup noodle_internal
 * @param buffer Buffer to fill with zero.
 * @param n Number of float elements.
 */
void noodle_reset_buffer(float *buffer, uint16_t n);

/**
 * @brief Add bias to a square output map and apply the requested activation.
 * @ingroup noodle_internal
 *
 * ACT_RELU clamps negative values to zero. ACT_NONE leaves biased values
 * unchanged. Other activation values are ignored here.
 *
 * @param output Output map with `n * n` values.
 * @param bias Bias scalar.
 * @param n Map width and height.
 * @param act Activation to apply.
 * @return @p n.
 */
uint16_t noodle_do_bias_act(float *output, float bias, uint16_t n, Activation act);

/**
 * @brief Accumulate one 2D transpose-convolution plane.
 * @ingroup noodle_internal
 *
 * The input plane is `[W][W]`; the kernel is `[K][K]`; output is accumulated in
 * `[Vt][Vt]` order instead of cleared.
 *
 * For explicit padding, callers choose @p OP so
 * `Vt = (W - 1) * S - 2 * P + K + OP` matches the desired output width.
 * With `P == 65535`, SAME-style transpose output uses `Vt = W * S`.
 *
 * @param input Input plane.
 * @param kernel Kernel values.
 * @param K Kernel width.
 * @param W Input width and height.
 * @param output Output accumulator.
 * @param P Padding per side, or `65535` for SAME-style padding.
 * @param S Stride.
 * @param OP User-computed output padding for explicit padding.
 * @return Output width.
 */
uint16_t noodle_do_conv_transpose(float *input, const float *kernel, uint16_t K,
                                  uint16_t W, float *output, uint16_t P,
                                  uint16_t S, uint16_t OP);

/**
 * @brief Find the maximum value and its index in a vector.
 * @ingroup noodle_internal
 * @param input Input vector.
 * @param n Number of values.
 * @param max_val Receives the maximum value.
 * @param max_idx Receives the index of the maximum value.
 */
void noodle_find_max(float *input, uint16_t n, float &max_val, uint16_t &max_idx);

/**
 * @brief Split packed batch-normalization parameters into four arrays.
 * @ingroup noodle_internal
 *
 * @param bn_params Packed `[gamma[N]][beta[N]][mean[N]][var[N]]` values.
 * @param N Number of elements per parameter array.
 * @param gamma Receives pointer to gamma values.
 * @param beta Receives pointer to beta values.
 * @param mean Receives pointer to mean values.
 * @param var Receives pointer to variance values.
 */
void noodle_unpack_bn_params(const float *bn_params, uint16_t N,
                             const float **gamma, const float **beta,
                             const float **mean, const float **var);

/**
 * @brief Apply 1D batch normalization in place to a raw vector.
 * @ingroup noodle_internal
 *
 * `gamma`, `beta`, `mean`, and `var` each contain @p N values.
 *
 * @param x Vector updated in place.
 * @param N Number of vector elements.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p N.
 */
uint16_t noodle_bn1d(float *x, uint16_t N, const float *gamma,
                     const float *beta, const float *mean, const float *var,
                     float eps);

/**
 * @brief Apply 1D batch normalization from packed parameters.
 * @ingroup noodle_internal
 *
 * `bn_params` is packed as `[gamma[N]][beta[N]][mean[N]][var[N]]`.
 *
 * @param x Vector updated in place.
 * @param N Number of vector elements.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p N.
 */
uint16_t noodle_bn1d(float *x, uint16_t N, const float *bn_params, float eps);

/**
 * @brief Apply 1D batch normalization followed by ReLU in place.
 * @ingroup noodle_internal
 *
 * `gamma`, `beta`, `mean`, and `var` each contain @p N values.
 *
 * @param x Vector updated in place.
 * @param N Number of vector elements.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p N.
 */
uint16_t noodle_bn1d_relu(float *x, uint16_t N, const float *gamma,
                          const float *beta, const float *mean,
                          const float *var, float eps);

/**
 * @brief Apply packed 1D batch normalization followed by ReLU in place.
 * @ingroup noodle_internal
 *
 * `bn_params` is packed as `[gamma[N]][beta[N]][mean[N]][var[N]]`.
 *
 * @param x Vector updated in place.
 * @param N Number of vector elements.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p N.
 */
uint16_t noodle_bn1d_relu(float *x, uint16_t N, const float *bn_params, float eps);

/**
 * @brief Apply 2D channel-wise batch normalization in place.
 * @ingroup noodle_internal
 *
 * Treats @p x as packed `[C][W][W]`. The parameter arrays each contain @p C
 * values, one per channel.
 *
 * @param x Tensor updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W.
 */
uint16_t noodle_bn2d(float *x, uint16_t C, uint16_t W, const float *gamma,
                     const float *beta, const float *mean, const float *var,
                     float eps);

/**
 * @brief Apply 2D channel-wise batch normalization from packed parameters.
 * @ingroup noodle_internal
 *
 * Treats @p x as packed `[C][W][W]`. `bn_params` is packed as
 * `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W.
 */
uint16_t noodle_bn2d(float *x, uint16_t C, uint16_t W, const float *bn_params,
                     float eps);

/**
 * @brief Apply 2D channel-wise batch normalization followed by ReLU.
 * @ingroup noodle_internal
 *
 * Treats @p x as packed `[C][W][W]`. The parameter arrays each contain @p C
 * values, one per channel.
 *
 * @param x Tensor updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W.
 */
uint16_t noodle_bn2d_relu(float *x, uint16_t C, uint16_t W,
                          const float *gamma, const float *beta,
                          const float *mean, const float *var, float eps);

/**
 * @brief Apply packed 2D batch normalization followed by ReLU.
 * @ingroup noodle_internal
 *
 * Treats @p x as packed `[C][W][W]`. `bn_params` is packed as
 * `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W.
 */
uint16_t noodle_bn2d_relu(float *x, uint16_t C, uint16_t W,
                          const float *bn_params, float eps);

/**
 * @brief Compute 2D convolution output width.
 * @ingroup noodle_internal
 * @param K Kernel width.
 * @param W Input width.
 * @param P Padding per side, or `65535` for SAME-style padding.
 * @param S Stride.
 * @return Output width.
 */
uint16_t noodle_compute_V(uint16_t K, uint16_t W, uint16_t P, uint16_t S);

/**
 * @brief Compute 2D convolution output width and effective asymmetric padding.
 * @ingroup noodle_internal
 * @param K Kernel width.
 * @param W Input width.
 * @param P Padding per side, or `65535` for SAME-style padding.
 * @param S Stride.
 * @param P0 Receives top/left padding.
 * @param P1 Receives bottom/right padding.
 * @return Output width.
 */
uint16_t noodle_compute_V_and_P(uint16_t K, uint16_t W, uint16_t P,
                                uint16_t S, uint16_t &P0, uint16_t &P1);

/**
 * @brief Apply valid max pooling to a packed channel-first tensor in place.
 * @ingroup noodle_internal
 * @param inplace Tensor in packed `[C][W][W]` order; compacted in place.
 * @param W Input width and height.
 * @param C Number of channels.
 * @param pool Pooling parameters.
 * @return Output width, or 0 for invalid parameters.
 */
uint16_t noodle_valid_max_pool(float *inplace, uint16_t W, uint16_t C,
                               const Pool &pool);

/**
 * @brief Compute transpose-convolution output width.
 * @ingroup noodle_internal
 *
 * For explicit padding, this computes
 * `V = (W - 1) * S - 2 * P + K + OP`; callers choose @p OP to match the
 * desired output width. With `P == 65535`, SAME-style output uses `V = W * S`.
 *
 * @param K Kernel width.
 * @param W Input width.
 * @param P Padding per side, or `65535` for SAME-style padding.
 * @param S Stride.
 * @param OP User-computed output padding for explicit padding.
 * @return Output width.
 */
uint16_t noodle_compute_Vt(uint16_t K, uint16_t W, uint16_t P, uint16_t S,
                           uint16_t OP);

/**
 * @brief Compute transpose-convolution output width and effective padding.
 * @ingroup noodle_internal
 *
 * For explicit padding, this computes
 * `V = (W - 1) * S - 2 * P + K + OP`; callers choose @p OP to match the
 * desired output width. With `P == 65535`, SAME-style output uses `V = W * S`
 * and derives asymmetric crop values.
 *
 * @param K Kernel width.
 * @param W Input width.
 * @param P Padding per side, or `65535` for SAME-style padding.
 * @param S Stride.
 * @param OP User-computed output padding for explicit padding.
 * @param P0 Receives top/left padding.
 * @param P1 Receives bottom/right padding.
 * @return Output width.
 */
uint16_t noodle_compute_Vt_and_P(uint16_t K, uint16_t W, uint16_t P,
                                 uint16_t S, uint16_t OP, uint16_t &P0,
                                 uint16_t &P1);

// ============================================================
// Private raw RAM/mixed layer implementations
// ============================================================

/**
 * @brief Raw memory-to-memory 1D convolution without pooling.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv1d(float *in, uint16_t n_inputs,
                       float *out, uint16_t n_outputs,
                       uint16_t W, const ConvMem &conv,
                       CBFPtr progress_cb);

/**
 * @brief Raw memory-to-memory 1D convolution with pooling.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv1d(float *in, uint16_t n_inputs,
                       float *out, uint16_t n_outputs,
                       uint16_t W, const ConvMem &conv,
                       const Pool &pool,
                       CBFPtr progress_cb);

/**
 * @brief Raw memory-to-file 1D convolution without pooling.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv1d(float *in, uint16_t n_inputs,
                       const char *out_fn, uint16_t n_outputs,
                       uint16_t W, const ConvMem &conv,
                       CBFPtr progress_cb);

/**
 * @brief Raw file-to-memory 1D convolution without pooling.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv1d(const char *in_fn, uint16_t n_inputs,
                       float *out, uint16_t n_outputs,
                       uint16_t W, const ConvMem &conv,
                       CBFPtr progress_cb);

/**
 * @brief File-to-memory 2D convolution with file-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv_float(const char *in_fn, uint16_t n_inputs,
                           uint16_t n_outputs, float *output,
                           uint16_t W, const Conv &conv,
                           const Pool &pool, CBFPtr progress_cb);

/**
 * @brief Memory-to-file 2D convolution with file-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv_float(float *input, uint16_t n_inputs,
                           uint16_t n_outputs, const char *out_fn,
                           uint16_t W, const Conv &conv,
                           const Pool &pool, CBFPtr progress_cb);

/**
 * @brief Memory-to-file 2D convolution with memory-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv_float(float *input, uint16_t n_inputs,
                           uint16_t n_outputs, const char *out_fn,
                           uint16_t W, const ConvMem &conv,
                           const Pool &pool, CBFPtr progress_cb);

/**
 * @brief Raw memory-to-memory 2D convolution with file-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv_float(float *input, uint16_t n_inputs,
                           uint16_t n_outputs, float *output,
                           uint16_t W, const Conv &conv,
                           const Pool &pool, CBFPtr progress_cb);

/**
 * @brief Raw memory-to-memory 2D convolution with memory-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv_float(float *input, uint16_t n_inputs,
                           uint16_t n_outputs, float *output,
                           uint16_t W, const ConvMem &conv,
                           const Pool &pool, CBFPtr progress_cb);

/**
 * @brief Raw memory-to-memory 2D convolution with near-PROGMEM parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_conv_float(float *input, uint16_t n_inputs,
                           uint16_t n_outputs, float *output,
                           uint16_t W, const ConvProgmem &conv,
                           const Pool &pool, CBFPtr progress_cb);

/**
 * @brief Raw memory-to-memory 2D transpose convolution.
 * @ingroup noodle_internal
 *
 * Input and output use packed channel-first layout. `conv.weight` is packed as
 * `[O][I][K][K]`. For explicit padding, callers set `conv.OP` so
 * `V = (W - 1) * conv.S - 2 * conv.P + conv.K + conv.OP` matches the desired
 * output width.
 *
 * @param input Packed `[I][W][W]` input tensor.
 * @param n_inputs Number of input channels.
 * @param n_outputs Number of output channels.
 * @param output Packed `[O][V][V]` output tensor.
 * @param W Input width and height.
 * @param conv Memory-backed transpose convolution parameters.
 * @param progress_cb Optional progress callback.
 * @return Output width, or 0 on null input/output/weights or invalid shape.
 */
uint16_t noodle_conv_transpose_float(float *input,
                                     uint16_t n_inputs,
                                     uint16_t n_outputs,
                                     float *output,
                                     uint16_t W,
                                     const ConvMem &conv,
                                     CBFPtr progress_cb);

/**
 * @brief Raw memory-to-memory depthwise convolution with file-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_dwconv_float(float *input, uint16_t n_channels,
                             float *output, uint16_t W,
                             const Conv &conv, const Pool &pool,
                             CBFPtr progress_cb);

/**
 * @brief Raw memory-to-memory depthwise convolution with memory-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_dwconv_float(float *input, uint16_t n_channels,
                             float *output, uint16_t W,
                             const ConvMem &conv, const Pool &pool,
                             CBFPtr progress_cb);

/**
 * @brief Raw memory-to-memory depthwise convolution with near-PROGMEM parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_dwconv_float(float *input, uint16_t n_channels,
                             float *output, uint16_t W,
                             const ConvProgmem &conv, const Pool &pool,
                             CBFPtr progress_cb);

/**
 * @brief Byte-input fully connected layer with file-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_fcn(const byte *input, uint16_t n_inputs,
                    uint16_t n_outputs, float *output,
                    const FCNFile &fcn, CBFPtr progress_cb);

/**
 * @brief Int8-input fully connected layer with file-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_fcn(const int8_t *input, uint16_t n_inputs,
                    uint16_t n_outputs, float *output,
                    const FCNFile &fcn, CBFPtr progress_cb);

/**
 * @brief Float-input fully connected layer with file-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs,
                    uint16_t n_outputs, float *output,
                    const FCNFile &fcn, CBFPtr progress_cb);

/**
 * @brief Float-input fully connected layer that writes output to a file.
 * @ingroup noodle_internal
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs,
                    uint16_t n_outputs, const char *out_fn,
                    const FCNFile &fcn, CBFPtr progress_cb);

/**
 * @brief File-input fully connected layer that writes output to memory.
 * @ingroup noodle_internal
 */
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs,
                    uint16_t n_outputs, float *output,
                    const FCNFile &fcn, CBFPtr progress_cb);

/**
 * @brief Float-input fully connected layer with memory-backed parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs,
                    uint16_t n_outputs, float *output,
                    const FCNMem &fcn, CBFPtr progress_cb);

/**
 * @brief Float-input fully connected layer with far-PROGMEM parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_fcn(const float *input, uint16_t n_inputs,
                    uint16_t n_outputs, float *output,
                    const FCNProgmem &fcn, CBFPtr progress_cb);

/**
 * @brief Float-input fully connected layer with near-PROGMEM parameters.
 * @ingroup noodle_internal
 */
uint16_t noodle_fcn_progmem(const float *input, uint16_t n_inputs,
                            uint16_t n_outputs, float *output,
                            const float *weight, const float *bias,
                            Activation act, CBFPtr progress_cb);

/**
 * @brief Write a float array to an already-open file.
 * @ingroup noodle_internal
 */
void noodle_array_to_file(float *array, NDL_File &fo, uint16_t n);

/**
 * @brief Write a byte grid to an already-open file.
 * @ingroup noodle_internal
 */
void noodle_grid_to_file(byte *grid, NDL_File &fo, uint16_t n);

/**
 * @brief Write a float grid to an already-open file.
 * @ingroup noodle_internal
 */
void noodle_grid_to_file(float *grid, NDL_File &fo, uint16_t n);

/**
 * @brief Read a float array from an already-open file.
 * @ingroup noodle_internal
 */
void noodle_array_from_file(NDL_File &fi, float *buffer, uint16_t K);

/**
 * @brief Read a byte grid from an already-open file.
 * @ingroup noodle_internal
 */
void noodle_grid_from_file(NDL_File &fi, byte *buffer, uint16_t K);

/**
 * @brief Read an int8 grid from an already-open file.
 * @ingroup noodle_internal
 */
void noodle_grid_from_file(NDL_File &fi, int8_t *buffer, uint16_t K);

/**
 * @brief Read a float grid from an already-open file.
 * @ingroup noodle_internal
 */
void noodle_grid_from_file(NDL_File &fi, float *buffer, uint16_t K);

/**
 * @brief Copy one square kernel from near-PROGMEM into RAM.
 * @ingroup noodle_internal
 *
 * @param w Base pointer to packed PROGMEM weights.
 * @param base Element offset of the first kernel value.
 * @param K Kernel width.
 * @param kernel Destination buffer with room for `K * K` floats.
 */
void noodle_copy_kernel_progmem(const float *w, uint32_t base,
                                uint16_t K, float *kernel);


// ============================================================
// Raw tensor utilities and activations
// ============================================================
// These raw-pointer functions are implementation-facing. User sketches should
// prefer the NoodleBuffer overloads declared in noodle.h.

/**
 * @brief Flatten a packed file tensor into an HWC-like raw vector.
 * @ingroup noodle_internal
 *
 * Reads packed `[C][V][V]` input and writes
 * `output[pixel * n_filters + channel]`.
 *
 * @param in_fn Input file containing packed channel-first planes.
 * @param output Destination buffer with room for `V * V * n_filters` floats.
 * @param V Input plane width and height.
 * @param n_filters Number of channel planes.
 * @return Number of floats written.
 */
uint16_t noodle_flat(const char *in_fn, float *output,
                     uint16_t V, uint16_t n_filters);

/**
 * @brief Flatten a packed memory tensor into an HWC-like raw vector.
 * @ingroup noodle_internal
 *
 * Reads @p input as packed `[C][V][V]` and writes
 * `output[pixel * n_filters + channel]`.
 *
 * @param input Source buffer with packed channel-first planes.
 * @param output Destination buffer with room for `V * V * n_filters` floats.
 * @param V Input plane width and height.
 * @param n_filters Number of channel planes.
 * @return Number of floats written.
 */
uint16_t noodle_flat(float *input, float *output,
                     uint16_t V, uint16_t n_filters);

/**
 * @brief Convert HWC-like raw data to packed channel-first raw data.
 * @ingroup noodle_internal
 *
 * Reads `src_hwc[pixel * C + channel]` and writes packed `[C][W][W]` data into
 * @p dst_chw.
 *
 * @param src_hwc Source buffer in HWC-like order.
 * @param dst_chw Destination buffer with room for `W * W * C` floats.
 * @param W Output plane width and height.
 * @param C Number of channel planes.
 * @return Number of floats written.
 */
uint16_t noodle_reshape(const float *src_hwc, float *dst_chw,
                        uint16_t W, uint16_t C);

/**
 * @brief Apply global average pooling in place to packed channel-first maps.
 * @ingroup noodle_internal
 *
 * Reduces `[C][W][W]` to `[C]` by writing each channel mean into the first
 * `C` positions of @p inout.
 *
 * @param inout Buffer containing packed `[C][W][W]` data.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @return @p C.
 */
uint16_t noodle_gap(float *inout, uint16_t C, uint16_t W);

/**
 * @brief Apply global max pooling in place to packed channel-first data.
 * @ingroup noodle_internal
 *
 * Reduces each channel to one maximum value in the first `C` positions of
 * @p inout. The current helper scans @p W values per channel.
 *
 * @param inout Buffer containing packed channel data.
 * @param C Number of channels.
 * @param W Number of values scanned per channel.
 * @return @p C.
 */
uint16_t noodle_gmp(float *inout, uint16_t C, uint16_t W);

/**
 * @brief Apply numerically stabilized softmax in place.
 * @ingroup noodle_internal
 * @param input_output Vector updated in place.
 * @param n Number of vector elements.
 * @return @p n.
 */
uint16_t noodle_soft_max(float *input_output, uint16_t n);

/**
 * @brief Apply sigmoid in place.
 * @ingroup noodle_internal
 * @param input_output Vector updated in place.
 * @param n Number of vector elements.
 * @return @p n.
 */
uint16_t noodle_sigmoid(float *input_output, uint16_t n);

/**
 * @brief Compute sigmoid for one scalar.
 * @ingroup noodle_internal
 * @param x Scalar input.
 * @return Logistic sigmoid of @p x.
 */
float noodle_sigmoidf(float x);

/**
 * @brief Apply logistic sigmoid in place.
 * @ingroup noodle_internal
 * @param input_output Vector updated in place.
 * @param n Number of vector elements.
 * @return @p n.
 */
uint16_t noodle_logit(float *input_output, uint16_t n);

/**
 * @brief Apply ReLU in place.
 * @ingroup noodle_internal
 * @param input_output Vector updated in place.
 * @param n Number of vector elements.
 * @return @p n.
 */
uint16_t noodle_relu(float *input_output, uint16_t n);

/**
 * @brief Backward-compatible raw alias for noodle_bn2d().
 * @ingroup noodle_internal
 *
 * Treats @p x as packed `[C][W][W]`.
 *
 * @param x Tensor updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W.
 */
uint16_t noodle_bn(float *x, uint16_t C, uint16_t W,
                   const float *gamma, const float *beta,
                   const float *mean, const float *var, float eps);

/**
 * @brief Backward-compatible raw alias for packed-parameter noodle_bn2d().
 * @ingroup noodle_internal
 *
 * Treats @p x as packed `[C][W][W]`. `bn_params` is packed as
 * `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W.
 */
uint16_t noodle_bn(float *x, uint16_t C, uint16_t W,
                   const float *bn_params, float eps);

/**
 * @brief Backward-compatible raw alias for noodle_bn2d_relu().
 * @ingroup noodle_internal
 *
 * Treats @p x as packed `[C][W][W]`.
 *
 * @param x Tensor updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param gamma Scale parameters.
 * @param beta Offset parameters.
 * @param mean Moving-mean parameters.
 * @param var Moving-variance parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W.
 */
uint16_t noodle_bn_relu(float *x, uint16_t C, uint16_t W,
                        const float *gamma, const float *beta,
                        const float *mean, const float *var, float eps);

/**
 * @brief Backward-compatible raw alias for packed-parameter noodle_bn2d_relu().
 * @ingroup noodle_internal
 *
 * Treats @p x as packed `[C][W][W]`. `bn_params` is packed as
 * `[gamma[C]][beta[C]][mean[C]][var[C]]`.
 *
 * @param x Tensor updated in place.
 * @param C Number of channels.
 * @param W Plane width and height.
 * @param bn_params Packed batch-normalization parameters.
 * @param eps Small value added to variance before inversion.
 * @return @p W.
 */
uint16_t noodle_bn_relu(float *x, uint16_t C, uint16_t W,
                        const float *bn_params, float eps);
