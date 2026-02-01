#pragma once

// Filesystem backend selection (exactly one)
// If the user didn't pick anything (including NONE), pick a default.
#if !defined(NOODLE_USE_SD_MMC) && !defined(NOODLE_USE_SDFAT) && !defined(NOODLE_USE_FFAT) && !defined(NOODLE_USE_LITTLEFS) && !defined(NOODLE_USE_NONE)
  #define NOODLE_USE_SDFAT
#endif


// Pooling enums
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
#else
  #error "invalid NOODLE_POOL_MODE"
#endif

#ifndef NOODLE_MAX_K
  #define NOODLE_MAX_K 5   // set 3 or 5 (or 7 if you really need)
#endif