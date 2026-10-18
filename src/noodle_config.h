#pragma once

// Numerical mode selection. Existing applications remain float32 by default.
// Define NOODLE_USE_INT8 before including noodle.h for a full-integer build.
#if defined(NOODLE_USE_FLOAT32) && defined(NOODLE_USE_INT8)
  #error "Choose only one Noodle numerical mode"
#endif
#if !defined(NOODLE_USE_FLOAT32) && !defined(NOODLE_USE_INT8)
  #define NOODLE_USE_FLOAT32
#endif
#if defined(NOODLE_USE_INT8) && defined(NOODLE_USE_Q8_WEIGHTS)
  #error "NOODLE_USE_Q8_WEIGHTS is a legacy float-activation mode; do not combine it with NOODLE_USE_INT8"
#endif
#if defined(NOODLE_USE_INT8)
  #pragma message "Noodle numerical mode = INT8"
#else
  #pragma message "Noodle numerical mode = FLOAT32"
#endif

// Filesystem backend selection (exactly one)
// If the user didn't pick anything (including NONE), pick a default.
#if !defined(NOODLE_USE_SD_MMC) && !defined(NOODLE_USE_SDFAT) && !defined(NOODLE_USE_FFAT) && !defined(NOODLE_USE_LITTLEFS) && !defined(NOODLE_USE_NONE)
  #define NOODLE_USE_SDFAT
#endif


// Weight storage selection
// When NOODLE_USE_Q8_WEIGHTS is defined, file-backed and memory-backed
// Conv/FCN weight arrays are stored as signed int8 values. Noodle dequantizes
// each weight to float during computation using: w = dq_scale * (q - dq_zp).
// Activations, accumulation, biases, and outputs remain float.
// Leave undefined for the original float32 weight storage.


// File scalar format selection
// TEXT: ASCII numeric values, one scalar per line.
// BIN : raw little-endian scalar values. float = IEEE-754 float32, byte = uint8_t.
#ifndef NOODLE_FILE_FORMAT_TEXT
  #define NOODLE_FILE_FORMAT_TEXT 0
#endif
#ifndef NOODLE_FILE_FORMAT_BIN
  #define NOODLE_FILE_FORMAT_BIN  1
#endif

#ifndef NOODLE_FILE_FORMAT
  #define NOODLE_FILE_FORMAT NOODLE_FILE_FORMAT_BIN
#endif

#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  #pragma message "file format = BIN"
#elif NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_TEXT
  #pragma message "file format = TEXT"
#else
  #error "invalid NOODLE_FILE_FORMAT"
#endif


// Pooling enums
#ifndef NOODLE_POOL_NONE
  #define NOODLE_POOL_NONE  0
#endif
#ifndef NOODLE_POOL_MAX
  #define NOODLE_POOL_MAX   1
#endif
#ifndef NOODLE_POOL_MEAN
  #define NOODLE_POOL_MEAN  2
#endif

// Select pooling mode
#ifndef NOODLE_POOL_MODE
  #define NOODLE_POOL_MODE NOODLE_POOL_MEAN
#endif

#if NOODLE_POOL_MODE == NOODLE_POOL_MAX
  #pragma message "pooling mode = MAX"
#elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
  #pragma message "pooling mode = MEAN"
#elif NOODLE_POOL_MODE == NOODLE_POOL_NONE
  #pragma message "pooling mode = NONE"
#else
  #error "invalid NOODLE_POOL_MODE"
#endif


#ifndef NOODLE_FCN_BLOCK
  /**
   * @brief Number of float weights buffered per file-backed FCN read block.
   *
   * Used by float-input fully connected layers with @ref FCNFile parameters.
   * Binary file mode reads each block as raw float32 data; text file mode fills
   * the same block by parsing scalar values one at a time. Higher values can
   * reduce file read calls, but increase stack usage by
   * `NOODLE_FCN_BLOCK * sizeof(float)` bytes.
   */
  #define NOODLE_FCN_BLOCK 128
#endif

// INT8 FCN parameter preload: weights per read in one bounded block.
// RAM-input FCNFile/FCNProgmem uses one buffer of this many bytes;
// SD-input FCNs use two buffers (input + weights), or 2 * this value.
// This does not change the Float32 NOODLE_FCN_BLOCK setting.
#ifndef NOODLE_INT8_FCN_BLOCK
  // Keep the earlier internal file-input tuning macro functional.
  #ifdef NOODLE_INT8_FILE_BLOCK
    #define NOODLE_INT8_FCN_BLOCK NOODLE_INT8_FILE_BLOCK
  #else
    #define NOODLE_INT8_FCN_BLOCK 128
  #endif
#endif
#if NOODLE_INT8_FCN_BLOCK < 1
  #error "NOODLE_INT8_FCN_BLOCK must be positive"
#endif

#ifndef NOODLE_MAX_K
  /**
   * @brief Largest convolution kernel width copied into fixed stack scratch.
   *
   * Convolution and depthwise-convolution paths use stack arrays sized from this
   * macro when copying one kernel at a time. Set it to the largest `K` used by
   * the firmware.
   */
  #define NOODLE_MAX_K 5
#endif

// INT8 2D convolution implementation selector.
// 0 keeps the original direct output-pixel kernel.
// 1 enables the tiled spatial-streaming 3x3 kernel; unsupported K values
//    automatically fall back to the direct implementation.
#ifndef NOODLE_INT8_CONV_IMPL_DIRECT
  #define NOODLE_INT8_CONV_IMPL_DIRECT 0
#endif
#ifndef NOODLE_INT8_CONV_IMPL_STREAMING
  #define NOODLE_INT8_CONV_IMPL_STREAMING 1
#endif
#ifndef NOODLE_INT8_CONV_IMPL
  #define NOODLE_INT8_CONV_IMPL NOODLE_INT8_CONV_IMPL_DIRECT
#endif

// Spatial output tile edge used by the INT8 streaming kernel.  This is not
// the convolution kernel width; a 3x3 layer still uses a 3x3 kernel.
#ifndef NOODLE_INT8_OUTPUT_TILE
  #define NOODLE_INT8_OUTPUT_TILE 8
#endif

// INT8 pointwise (1x1) convolution implementation selector.
// 0 keeps the ordinary one-output-channel path.
// 1 blocks output channels so each input value is reused across a small
//    group of pointwise filters.
#ifndef NOODLE_INT8_PW_IMPL
  #define NOODLE_INT8_PW_IMPL 0
#endif
#ifndef NOODLE_INT8_PW_OUTPUT_BLOCK
  #define NOODLE_INT8_PW_OUTPUT_BLOCK 8
#endif

// Hidden global NoodleBuffer arena.
// The first noodle_buffer_init() lazily allocates this many bytes.
// Logical buffers start at zero capacity and grow transparently.
#ifndef NOODLE_BUFFER_ARENA_INITIAL_BYTES
  #ifdef NOODLE_BUFFER_ARENA_INITIAL_FLOATS
    // Backward compatibility with existing build flags.
    #define NOODLE_BUFFER_ARENA_INITIAL_BYTES \
      ((NOODLE_BUFFER_ARENA_INITIAL_FLOATS) * sizeof(float))
  #else
    #define NOODLE_BUFFER_ARENA_INITIAL_BYTES 64u
  #endif
#endif


