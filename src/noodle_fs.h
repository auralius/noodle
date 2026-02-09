/**
 * @file noodle_fs.h
 * @defgroup noodle_fs Filesystem backend layer
 * @ingroup noodle_fs
 *
 * Backend selector for Noodle: SdFat, SD_MMC, FFat, LittleFS, or NONE.
 * Noodle supports multiple storage backends (SdFat, SD_MMC, FFat, LittleFS, or NONE).
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
 *   - other file operations
 *
 *
 * @par Path normalization
 * Some Arduino filesystem APIs require paths to start with a leading slash (e.g. "/w01.txt"),
 * while SdFat commonly uses bare names (e.g. "w01.txt"). The helper
 * noodle_norm_filename() applies this rule.
 *
 * @warning noodle_norm_filename() returns a pointer to a static buffer when a leading slash is needed.
 *          That means it is **not re-entrant** and **not thread-safe**: do not store the returned pointer;
 *          use it immediately for open/remove.
 */


#pragma once

// -----------------------------
// Path policy 
// -----------------------------
#if defined(NOODLE_USE_SDFAT)
  #define NOODLE_FS_NEEDS_LEADING_SLASH 0
#else
  #define NOODLE_FS_NEEDS_LEADING_SLASH 1
#endif

#ifndef NOODLE_MAX_FILENAME
  #define NOODLE_MAX_FILENAME 20   // plenty for "w28.txt" etc.
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

    #if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
      using NDL_File = FsFile;   // SdFat on ESP32 typically uses FsFile
    #else
      using NDL_File = File;     // SdFat on AVR often uses File wrapper
    #endif

    extern SdFat NOODLE_FS;      // defined in noodle.cpp

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
// Safe string copy with NUL-termination

// Minimal bounded copy for short filenames
/** @brief Copy a C string into a bounded buffer with NUL-termination. */

inline void noodle_copy_name(char* dst, size_t cap, const char* src) {
  if (!dst || cap == 0) return;
  if (!src) { dst[0] = '\0'; return; }

  size_t i = 0;
  while (src[i] != '\0' && (i + 1) < cap) {
    dst[i] = src[i];
    i++;
  }
  dst[i] = '\0';
}

// Normalize path according to backend policy
// Only safe for immediate use (open/remove); do not store returned pointer!
/**
 * @brief Normalize a filename/path for the selected filesystem backend.
 *
 * For backends that require a leading '/', this function prepends it.
 * For SdFat, the input is returned as-is.
 *
 * @param name Input filename, e.g. "w01.txt".
 * @return Pointer to a normalized path string.
 * @warning For slash-requiring backends, the returned pointer refers to a static buffer.
 */

inline const char* noodle_norm_filename(const char* name) {
  if (!name) return "";

#if NOODLE_FS_NEEDS_LEADING_SLASH
  // One static buffer is enough for your simplified rules.
  // Size = '/' + filename + '\0'
  static char out[NOODLE_MAX_FILENAME + 2];

  out[0] = '/';
  noodle_copy_name(out + 1, NOODLE_MAX_FILENAME + 1, name);
  return out;
#else
  // SdFat: return as-is, no extra RAM used.
  return name;
#endif
}

// Open a file for reading
/** @brief Open a file for reading using the selected backend. */

inline NDL_File noodle_fs_open_read(const char* path) {
  path = noodle_norm_filename(path);
#if defined(NOODLE_USE_NONE)
  (void)path;
  return NDL_File{}; // invalid handle
#elif defined(NOODLE_USE_SDFAT)
  return NOODLE_FS.open(path, O_RDONLY);
#else
  return NOODLE_FS.open(path, FILE_READ);
#endif
}

// Open a file for writing (creates or truncates)
/** @brief Open a file for writing (create or truncate) using the selected backend. */

inline NDL_File noodle_fs_open_write(const char* path) {
    path = noodle_norm_filename(path);
#if defined(NOODLE_USE_NONE)
  (void)path;
  return NDL_File{}; // invalid handle
#elif defined(NOODLE_USE_SDFAT)
  int flags = O_WRITE | O_CREAT | O_TRUNC;
  return NOODLE_FS.open(path, flags);
#else
  return NOODLE_FS.open(path, FILE_WRITE);
#endif
}

/** @brief Remove a file using the selected backend. */


inline bool noodle_fs_remove(const char* path) {
    path = noodle_norm_filename(path);
#if defined(NOODLE_USE_NONE)
  (void)path;
  return false;
#else
  return NOODLE_FS.remove(path);
#endif
}

/**
 * @brief Rewind a file handle to position 0.
 *
 * Different backends expose different seek APIs (seekSet vs seek).
 */


inline void noodle_rewind_file(NDL_File &fi) {
#if defined(NOODLE_USE_SDFAT)
  fi.seekSet(0);
#elif defined(NOODLE_USE_FFAT) || defined(NOODLE_USE_LITTLEFS) || defined(NOODLE_USE_SD_MMC) || defined(NOODLE_USE_SD)
  fi.seek(0);
#else
  // NOODLE_USE_NONE (or unknown backend): nothing to seek. Close the handle.
  fi.close();
#endif
}
