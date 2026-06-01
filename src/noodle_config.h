#pragma once

// Filesystem backend selection (exactly one)
// If the user didn't pick anything (including NONE), pick a default.
#if !defined(NOODLE_USE_SD_MMC) && !defined(NOODLE_USE_SDFAT) && !defined(NOODLE_USE_FFAT) && !defined(NOODLE_USE_LITTLEFS) && !defined(NOODLE_USE_NONE)
  #define NOODLE_USE_SDFAT
#endif



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

// Define maksimum kernel size to avoid variable allocation!
#ifndef NOODLE_MAX_K
  #define NOODLE_MAX_K 5 
#endif
