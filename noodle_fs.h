/**
 * @file noodle_fs.h
 * @brief Backend selector for Noodle: SdFat, SD_MMC, or FFat.
 *
 * Define exactly one of:
 *   - NOODLE_USE_SDFAT
 *   - NOODLE_USE_SD_MMC 
 *   - NOODLE_USE_FFAT
 *
 * Exposes:
 *   - NDL_File (FsFile for SdFat, File otherwise)
 *   - NOODLE_FS (SdFat/SD_MMC/FFat object)
 *   - noodle_fs_open_read / noodle_fs_open_write / noodle_fs_remove
 */
#pragma once

// Optional user override header (lives in the sketch folder)
#if defined(__has_include)
#  if __has_include("noodle_fs_backend.h")
#    include "noodle_fs_backend.h"
#  endif
#endif

// Fallback default ONLY if nothing was chosen by the user
#if !defined(NOODLE_USE_SDFAT) && !defined(NOODLE_USE_SD_MMC) && !defined(NOODLE_USE_FFAT)
  #define NOODLE_USE_SD_MMC   // pick your preferred default
#endif

#if defined(NOODLE_USE_SDFAT)
  #include <SdFat.h>
  extern SdFat NOODLE_FS;          // defined in noodle.cpp for SdFat builds
  using NDL_File = FsFile;

#elif defined(NOODLE_USE_FFAT)
  #include <FFat.h>
  #define NOODLE_FS FFat
  using NDL_File = File;

#elif defined(NOODLE_USE_SD_MMC)
  #include <SD_MMC.h>
  #define NOODLE_FS SD_MMC
  using NDL_File = File;
#endif

inline NDL_File noodle_fs_open_read(const char* path) {
#if defined(NOODLE_USE_SDFAT)
  return NOODLE_FS.open(path, O_RDONLY);
#else
  return NOODLE_FS.open(path, FILE_READ);
#endif
}

inline NDL_File noodle_fs_open_write(const char* path) {
#if defined(NOODLE_USE_SDFAT)
  uint8_t flags = O_WRITE | O_CREAT;
  return NOODLE_FS.open(path, flags);
#else
  return NOODLE_FS.open(path, FILE_WRITE);
#endif
}

inline bool noodle_fs_remove(const char* path) {
  return NOODLE_FS.remove(path);
}
