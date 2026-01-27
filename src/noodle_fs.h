/**
 * @file noodle_fs.h
 * @brief Backend selector for Noodle: SdFat, SD_MMC, FFat, LittleFS, or NONE.
 *
 * Define exactly ONE of:
 *   - NOODLE_USE_SDFAT
 *   - NOODLE_USE_SD_MMC
 *   - NOODLE_USE_FFAT
 *   - NOODLE_USE_LITTLEFS
 *   - NOODLE_USE_NONE        // <--- no filesystem / no storage
 *
 * Exposes:
 *   - NDL_File (FsFile for SdFat, File otherwise, or NDL_NullFile for NONE)
 *   - NOODLE_FS (SdFat object for SdFat; FFat/SD_MMC/LittleFS singleton otherwise; not used for NONE)
 *   - noodle_fs_open_read / noodle_fs_open_write / noodle_fs_remove
 */
#pragma once

#if defined(NOODLE_USE_SDFAT)
#warning "NOODLE_USE_SDFAT is defined"
#endif
#if defined(NOODLE_USE_SD_MMC)
#warning "NOODLE_USE_SD_MMC is defined"
#endif
#if defined(NOODLE_USE_FFAT)
#warning "NOODLE_USE_FFAT is defined"
#endif
#if defined(NOODLE_USE_LITTLEFS)
#warning "NOODLE_USE_LITTLEFS is defined"
#endif
#if defined(NOODLE_USE_NONE)
#warning "NOODLE_USE_NONE is defined"
#endif


// ------------------------------
// 1) Enforce "exactly one"
// ------------------------------
#if (defined(NOODLE_USE_SDFAT) + defined(NOODLE_USE_SD_MMC) + defined(NOODLE_USE_FFAT) + defined(NOODLE_USE_LITTLEFS) + defined(NOODLE_USE_NONE)) != 1
# error "Select exactly ONE backend: NOODLE_USE_SDFAT, NOODLE_USE_SD_MMC, NOODLE_USE_FFAT, NOODLE_USE_LITTLEFS, or NOODLE_USE_NONE"
#endif

// ------------------------------
// 2) NONE backend
// ------------------------------
#if defined(NOODLE_USE_NONE)

  // Minimal "file-like" stub so code compiles.
  struct NDL_NullFile {
    bool _open = false;

    // Match common File/FsFile-ish methods used in embedded code.
    operator bool() const { return _open; }
    bool available() const { return false; }
    int  read() { return -1; }
    int  peek() { return -1; }
    size_t readBytes(char*, size_t) { return 0; }
    size_t write(uint8_t) { return 0; }
    size_t write(const uint8_t*, size_t) { return 0; }
    void flush() {}
    void close() { _open = false; }
    size_t size() const { return 0; }
    size_t position() const { return 0; }
    bool seek(uint32_t) { return false; }
    size_t println(byte v) { (void)v; return 0; }
    size_t println(float v) { (void)v; return 0; }
    size_t println(float v, int base) { (void)v; return 0; }
  };

  using NDL_File = NDL_NullFile;

  // No NOODLE_FS symbol in NONE mode.

#else
// ------------------------------
// 3) Real backends
// ------------------------------

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
#endif

// ------------------------------
// 4) Unified API
// ------------------------------
inline NDL_File noodle_fs_open_read(const char* path) {
#if defined(NOODLE_USE_NONE)
  (void)path;
  return NDL_File{}; // invalid handle
#elif defined(NOODLE_USE_SDFAT)
  return NOODLE_FS.open(path, O_RDONLY);
#else
  return NOODLE_FS.open(path, FILE_READ);
#endif
}

inline NDL_File noodle_fs_open_write(const char* path) {
#if defined(NOODLE_USE_NONE)
  (void)path;
  return NDL_File{}; // invalid handle
#elif defined(NOODLE_USE_SDFAT)
  uint8_t flags = O_WRITE | O_CREAT | O_TRUNC;
  return NOODLE_FS.open(path, flags);
#else
  return NOODLE_FS.open(path, FILE_WRITE);
#endif
}

inline bool noodle_fs_remove(const char* path) {
#if defined(NOODLE_USE_NONE)
  (void)path;
  return false;
#else
  return NOODLE_FS.remove(path);
#endif
}
