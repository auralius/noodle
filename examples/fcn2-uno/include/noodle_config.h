// noodle_fs_backend.h  (in the sketch folder)
//#define NOODLE_USE_SD_MMC
// or
#define NOODLE_USE_SDFAT
// or
//#define NOODLE_USE_FFAT
// or
//#define NOODLE_USE_LITTLEFS

#define NOODLE_POOL_MAX   1
#define NOODLE_POOL_MEAN  2

// Select pooling mode
//#define NOODLE_POOL_MODE NOODLE_POOL_MAX
#define NOODLE_POOL_MODE NOODLE_POOL_MEAN

#if NOODLE_POOL_MODE == NOODLE_POOL_MAX
  #pragma message "pooling mode = MAX"
#elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
  #pragma message "pooling mode = MEAN"
#else
  #error "invalid NOODLE_POOL_MODE"
#endif 