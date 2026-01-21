/**
 * @file noodle_fs.h
 * @brief Backend selector for Noodle: SdFat, SD_MMC, FFat, or LittleFS.
 *
 * Define exactly ONE of:
 *   - NOODLE_USE_SDFAT
 *   - NOODLE_USE_SD_MMC
 *   - NOODLE_USE_FFAT
 *   - NOODLE_USE_LITTLEFS
 *
 * Exposes:
 *   - NDL_File (FsFile for SdFat, File otherwise)
 *   - NOODLE_FS (SdFat object for SdFat; FFat/SD_MMC/LittleFS singleton otherwise)
 *   - noodle_fs_open_read / noodle_fs_open_write / noodle_fs_remove
 */
#pragma once

#if !defined(NOODLE_USE_SDFAT) && !defined(NOODLE_USE_SD_MMC) && !defined(NOODLE_USE_FFAT) && !defined(NOODLE_USE_LITTLEFS)
# error "Select file backend! Define exactly one of NOODLE_USE_SDFAT, NOODLE_USE_SD_MMC, NOODLE_USE_FFAT, NOODLE_USE_LITTLEFS"
#endif

#if defined(NOODLE_USE_SDFAT)
  #include <SdFat.h>

  #if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
    using NDL_File = FsFile;   // SdFat on ESP32 typically uses FsFile
  #else
    using NDL_File = File;     // SdFat on AVR often uses File wrapper
  #endif

  extern SdFat NOODLE_FS;  // defined in noodle.cpp

#elif defined(NOODLE_USE_FFAT)
  #include <FFat.h>
  #define NOODLE_FS FFat
  using NDL_File = File;

#elif defined(NOODLE_USE_LITTLEFS)
  #include <LittleFS.h>
  #define NOODLE_FS LittleFS
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
