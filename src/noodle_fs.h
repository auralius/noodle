/**
 * @file noodle_fs.h
 * @defgroup noodle_fs Filesystem Backend Layer
 *
 * @brief Small compatibility layer over the storage backends used by Noodle.
 *
 * Noodle reads and writes model tensors through a tiny common API so the same
 * higher-level code can run on SdFat, SD_MMC, FFat, LittleFS, or with storage
 * disabled. This header selects the backend-specific includes and exposes:
 *
 * - @ref NDL_File, the file-handle type used by the public API.
 * - `NOODLE_FS`, the selected filesystem object or singleton for real backends.
 * - noodle_fs_open_read(), noodle_fs_open_write(), noodle_fs_remove(), and
 *   noodle_rewind_file().
 *
 * Define exactly one backend macro before this header is included:
 *
 * - `NOODLE_USE_SDFAT`
 * - `NOODLE_USE_SD_MMC`
 * - `NOODLE_USE_FFAT`
 * - `NOODLE_USE_LITTLEFS`
 * - `NOODLE_USE_NONE`
 *
 * When using the main `noodle.h` entry point, `noodle_config.h` is included
 * first and selects `NOODLE_USE_SDFAT` by default if no backend macro is set.
 * If this header is included directly, the caller must define one backend
 * macro first.
 *
 * @section noodle_fs_paths Path Normalization
 * Arduino filesystem APIs such as SD_MMC, FFat, and LittleFS expect paths with
 * a leading slash, for example `"/w01.txt"`. SdFat commonly uses bare names,
 * for example `"w01.txt"`. noodle_norm_filename() applies this policy before
 * open/remove calls.
 *
 * @warning For slash-requiring backends, noodle_norm_filename() returns a
 * pointer to a static buffer. It is not re-entrant or thread-safe. Use the
 * returned pointer immediately and do not store it.
 */


#pragma once

#include <stddef.h>
#include <stdint.h>

// -----------------------------
// Path policy
// -----------------------------
/**
 * @brief Whether the selected backend expects normalized paths to start with '/'.
 * @ingroup noodle_fs
 *
 * SdFat is treated as accepting bare filenames. Other real Arduino filesystem
 * backends are normalized to slash-prefixed paths.
 */
#if defined(NOODLE_USE_SDFAT)
  #define NOODLE_FS_NEEDS_LEADING_SLASH 0
#else
  #define NOODLE_FS_NEEDS_LEADING_SLASH 1
#endif

/**
 * @brief Maximum filename length copied by noodle_norm_filename().
 * @ingroup noodle_fs
 *
 * The limit applies to the input name characters, excluding any added leading
 * slash and trailing NUL. Longer names are truncated by noodle_copy_name().
 */
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

  /**
   * @brief Minimal invalid file handle used when filesystem support is disabled.
   * @ingroup noodle_fs
   *
   * This type implements the small File/FsFile method subset used by Noodle, but
   * all reads, writes, seeks, and size queries fail or return zero. It allows
   * code that references file APIs to compile under `NOODLE_USE_NONE`; runtime
   * file-backed operations should treat returned handles as invalid.
   */
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
    size_t println(uint8_t v) { (void)v; return 0; }
    size_t println(float v) { (void)v; return 0; }
    size_t println(float v, int base) { (void)v; return 0; }
  };

  /**
   * @brief File-handle type used by Noodle file APIs.
   * @ingroup noodle_fs
   */
  using NDL_File = NDL_NullFile;

  // No NOODLE_FS symbol in NONE mode.

#else
// ------------------------------
// 3) Real backends
// ------------------------------
  #if defined(NOODLE_USE_SDFAT)
    #include <SdFat.h>

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || \
    defined(ESP8266) || defined(ARDUINO_ARCH_ESP8266) || \
    defined(ARDUINO_ARCH_RP2040)
      using NDL_File = FsFile;   // SdFat on ESP32 typically uses FsFile
    #else
      using NDL_File = File;     // SdFat on AVR often uses File wrapper
    #endif

    /**
     * @brief Selected filesystem object for the SdFat backend.
     * @ingroup noodle_fs
     *
     * The object is declared here and defined in `noodle_internal.cpp`.
     */
    extern SdFat NOODLE_FS;      // defined in noodle_internal.cpp

  #elif defined(NOODLE_USE_FFAT)
    #include <FFat.h>
    /** @brief Selected filesystem singleton for the FFat backend. */
    #define NOODLE_FS FFat
    using NDL_File = File;

  #elif defined(NOODLE_USE_LITTLEFS)
    #include <LittleFS.h>
    /** @brief Selected filesystem singleton for the LittleFS backend. */
    #define NOODLE_FS LittleFS
    using NDL_File = File;

  #elif defined(NOODLE_USE_SD_MMC)
    #include <SD_MMC.h>
    /** @brief Selected filesystem singleton for the SD_MMC backend. */
    #define NOODLE_FS SD_MMC
    using NDL_File = File;

  #endif
#endif

// ------------------------------
// 4) Unified API
// ------------------------------
/**
 * @brief Copy a C string into a bounded buffer with NUL-termination.
 * @ingroup noodle_fs
 *
 * If @p src is `nullptr`, @p dst is set to an empty string. If @p src is longer
 * than the destination capacity, the result is truncated and still
 * NUL-terminated.
 *
 * @param dst Destination buffer.
 * @param cap Destination capacity in bytes, including the trailing NUL.
 * @param src Source C string.
 */
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

/**
 * @brief Normalize a filename/path for the selected filesystem backend.
 * @ingroup noodle_fs
 *
 * SdFat paths are returned unchanged. For backends that require a leading slash,
 * this function prepends `'/'` unless @p name already starts with one.
 *
 * @param name Input filename or path, for example `"w01.txt"` or `"/w01.txt"`.
 * @return Pointer to the normalized path string.
 * @warning For slash-requiring backends, the returned pointer refers to a static
 * buffer. Use it immediately and do not store it.
 */
inline const char* noodle_norm_filename(const char* name) {
  if (!name) return "";

#if NOODLE_FS_NEEDS_LEADING_SLASH
  // One static buffer is enough for your simplified rules.
  // Size = '/' + filename + '\0'
  static char out[NOODLE_MAX_FILENAME + 2];

  if (name[0] == '/') {
    noodle_copy_name(out, sizeof(out), name);
    return out;
  }

  out[0] = '/';
  noodle_copy_name(out + 1, NOODLE_MAX_FILENAME + 1, name);
  return out;
#else
  // SdFat: return as-is, no extra RAM used.
  return name;
#endif
}

/**
 * @brief Open a file for reading using the selected backend.
 * @ingroup noodle_fs
 *
 * The path is normalized with noodle_norm_filename() before opening. In
 * `NOODLE_USE_NONE` mode this returns an invalid `NDL_File`.
 *
 * @param path Filename or path.
 * @return Open file handle, or an invalid handle on failure.
 */
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

/**
 * @brief Open a file for writing using the selected backend.
 * @ingroup noodle_fs
 *
 * The path is normalized with noodle_norm_filename() before opening. Existing
 * SdFat files are truncated. Other backends use their `FILE_WRITE` mode. In
 * `NOODLE_USE_NONE` mode this returns an invalid `NDL_File`.
 *
 * @param path Filename or path.
 * @return Open file handle, or an invalid handle on failure.
 */
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

/**
 * @brief Remove a file using the selected backend.
 * @ingroup noodle_fs
 *
 * The path is normalized with noodle_norm_filename() before removal.
 *
 * @param path Filename or path.
 * @return `true` when the selected backend reports that the file was removed.
 */
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
 * @ingroup noodle_fs
 *
 * SdFat exposes `seekSet(0)`, while Arduino `File` backends expose `seek(0)`.
 * In `NOODLE_USE_NONE` mode the handle is closed.
 *
 * @param fi Open file handle to rewind.
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
