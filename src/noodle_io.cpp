/**
 * @file noodle_io.cpp
 * @brief Noodle file/storage I/O helpers.
 * @ingroup noodle_api
 */
#include "noodle_internal.h"

void noodle_read_top_line(const char* fn, char *line, size_t maxlen) {
  line[0] = '\0'; // empty by default

  NDL_File f = noodle_fs_open_read(fn);
  if (!f) {
    return;
  }

  size_t i = 0;
  while (f.available() && i < maxlen - 1) {
    char c = f.read();
    if (c == '\n' || c == '\r') {
      break; // stop at first newline
    }
    line[i++] = c;
  }
  line[i] = '\0'; // null-terminate
  f.close();
}

size_t noodle_read_bytes_until(NDL_File &file,
                               char terminator,
                               char *buffer,
                               size_t length) {
  size_t count = 0;
  int c;
  while (count < length - 1) {
    c = file.read();
    if (c < 0 || (char)c == terminator) break;
    buffer[count++] = (char)c;
  }
  buffer[count] = '\0';
  return count;
}

bool noodle_fs_init(uint8_t clk_pin,
                    uint8_t cmd_pin,
                    uint8_t d0_pin) {
#if defined(NOODLE_USE_NONE)
  (void)clk_pin; (void)cmd_pin; (void)d0_pin;
  return false;

#elif defined(NOODLE_USE_SD_MMC)
  // SD_MMC supports setPins on some ESP32 boards
  SD_MMC.setPins(clk_pin, cmd_pin, d0_pin);
  // 1-bit mode = true (since only D0 provided)
  return SD_MMC.begin("/sdcard", true, false, 20000, 5);

#else
  // Not SD_MMC: these pins are meaningless
  (void)clk_pin; (void)cmd_pin; (void)d0_pin;
  return noodle_fs_init();
#endif
}

bool noodle_fs_init(uint8_t clk_pin,
                    uint8_t cmd_pin,
                    uint8_t d0_pin,
                    uint8_t d1_pin,
                    uint8_t d2_pin,
                    uint8_t d3_pin) {
#if defined(NOODLE_USE_NONE)
  (void)clk_pin; (void)cmd_pin; (void)d0_pin; (void)d1_pin; (void)d2_pin; (void)d3_pin;
  return false;

#elif defined(NOODLE_USE_SD_MMC)
  // 4-bit SDMMC mode (D0..D3)
  // Arduino-ESP32 SD_MMC expects setPins(clk, cmd, d0, d1, d2, d3)
  SD_MMC.setPins(clk_pin, cmd_pin, d0_pin, d1_pin, d2_pin, d3_pin);

  // 1-bit mode = false (we are providing D1..D3)
  return SD_MMC.begin("/sdcard", false, false, 20000, 5);

#else
  // Not SD_MMC: these pins are meaningless
  (void)clk_pin; (void)cmd_pin; (void)d0_pin; (void)d1_pin; (void)d2_pin; (void)d3_pin;
  return noodle_fs_init();
#endif
}


bool noodle_fs_init() {
#if defined(NOODLE_USE_NONE)
  return false;

#elif defined(NOODLE_USE_SD_MMC)
  return SD_MMC.begin("/sdcard", false, false, 20000, 5);

#elif defined(NOODLE_USE_SDFAT)
  return NOODLE_FS.begin();

#else
  // FFat / LittleFS
  return NOODLE_FS.begin();
#endif
}

bool noodle_fs_init(uint8_t cs_pin) {
#if defined(NOODLE_USE_NONE)
  (void)cs_pin;
  return false;

#elif defined(NOODLE_USE_SDFAT)
  pinMode(cs_pin, OUTPUT);
  digitalWrite(cs_pin, HIGH);

  SdSpiConfig cfg(cs_pin, DEDICATED_SPI, SD_SCK_MHZ(4));
  return NOODLE_FS.begin(cfg);

#elif defined(NOODLE_USE_SD_MMC)
  (void)cs_pin;
  return SD_MMC.begin("/sdcard", false, false, 20000, 5);

#else
  // FFat / LittleFS ignore CS
  (void)cs_pin;
  return NOODLE_FS.begin();
#endif
}

#if defined(NOODLE_USE_SDFAT)
bool noodle_fs_init(uint8_t cs_pin, SPIClass &spi, uint8_t sck_mhz) {
  pinMode(cs_pin, OUTPUT);
  digitalWrite(cs_pin, HIGH);

  spi.begin();

  SdSpiConfig cfg(cs_pin, SHARED_SPI, SD_SCK_MHZ(sck_mhz), &spi);
  return NOODLE_FS.begin(cfg);
}
#endif

#if defined(NOODLE_USE_SD_MMC)
  #include "esp_vfs_fat.h"
  #include "sdmmc_cmd.h"
  #include "driver/sdmmc_host.h"
#endif

size_t noodle_read_raw(NDL_File &f, void *dst, size_t n) {
#if defined(NOODLE_USE_NONE)
  (void)f; (void)dst; (void)n;
  return 0;
#else
  return f.read((uint8_t *)dst, n);
#endif
}

size_t noodle_write_raw(NDL_File &f, const void *src, size_t n) {
#if defined(NOODLE_USE_NONE)
  (void)f; (void)src; (void)n;
  return 0;
#else
  return f.write((const uint8_t *)src, n);
#endif
}

size_t noodle_read_float_block(NDL_File &f, float *dst, size_t n_floats) {
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  const size_t want = n_floats * sizeof(float);
  const size_t got = noodle_read_raw(f, dst, want);
  return got / sizeof(float);
#else
  for (size_t i = 0; i < n_floats; i++) {
    dst[i] = noodle_read_float(f);
  }
  return n_floats;
#endif
}


float noodle_read_float(NDL_File &f) {
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  float v = 0.0f;
  const size_t n = noodle_read_raw(f, &v, sizeof(v));
  return (n == sizeof(v)) ? v : 0.0f;
#else
  char s[20];
  size_t n = noodle_read_bytes_until(f, '\n', (char *)s, sizeof(s));
  s[n] = '\0';
  return atof(s);
#endif
}

byte noodle_read_byte(NDL_File &f) {
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  uint8_t v = 0;
  const size_t n = noodle_read_raw(f, &v, sizeof(v));
  return (n == sizeof(v)) ? (byte)v : (byte)0;
#else
  char s[20];
  size_t n = noodle_read_bytes_until(f, '\n', (char *)s, sizeof(s));
  s[n] = '\0';
  return (byte)atoi(s);
#endif
}

void noodle_write_float(NDL_File &f,
                        float d) {
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  noodle_write_raw(f, &d, sizeof(d));
#else
  f.println(d, 6);
#endif
}

void noodle_write_byte(NDL_File &f,
                       byte d) {
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  const uint8_t v = (uint8_t)d;
  noodle_write_raw(f, &v, sizeof(v));
#else
  f.println(d);
#endif
}

void noodle_delete_file(const char *fn) {
  noodle_fs_remove(fn);
}

void noodle_array_to_file(float *array,
                          const char *fn,
                          uint16_t n) {
  fo = noodle_fs_open_write(fn);
  for (uint16_t i = 0; i < n; i++)
    noodle_write_float(fo, array[i]);
  fo.close();
}

void noodle_array_to_file(float *array,
                          NDL_File &fo,
                          uint16_t n) {
  for (uint16_t i = 0; i < n; i++)
    noodle_write_float(fo, array[i]);
}

void noodle_grid_to_file(byte *grid,
                         const char *fn,
                         uint16_t n) {
  fo = noodle_fs_open_write(fn);
  for (uint16_t i = 0; i < n; i++) {
    const uint16_t row = i * n;
    for (uint16_t j = 0; j < n; j++) {
      noodle_write_float(fo, (float)grid[row + j]);
    }
  }
  fo.close();
}

void noodle_grid_to_file(byte *grid,
                         NDL_File &fo,
                         uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    const uint16_t row = i * n;
    for (uint16_t j = 0; j < n; j++) {
      noodle_write_float(fo, (float)grid[row + j]);
    }
  }
}

void noodle_grid_to_file(float *grid,
                         const char *fn,
                         uint16_t n) {
  fo = noodle_fs_open_write(fn);
  for (uint16_t i = 0; i < n; i++) {
    const uint16_t row = i * n;
    for (uint16_t j = 0; j < n; j++) {
      noodle_write_float(fo, grid[row + j]);
    }
  }
  fo.close();
}

void noodle_grid_to_file(float *grid,
                         NDL_File &fo,
                         uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    const uint16_t row = i * n;
    for (uint16_t j = 0; j < n; j++) {
      noodle_write_float(fo, grid[row + j]);
    }
  }
}

// ===== Read helpers moved from noodle_conv.cpp =====

void noodle_array_from_file(const char *fn,
                            float *buffer,
                            uint16_t K) {
  fi = noodle_fs_open_read(fn);
  for (uint16_t i = 0; i < K; i++)
    buffer[i] = noodle_read_float(fi);
  fi.close();
}

void noodle_array_from_file(NDL_File &fi,
                            float *buffer,
                            uint16_t K) {
  for (uint16_t i = 0; i < K; i++)
    buffer[i] = noodle_read_float(fi);
}

void noodle_grid_from_file(const char *fn,
                           byte *buffer,
                           uint16_t K) {
  fi = noodle_fs_open_read(fn);
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[i * K + j] = noodle_read_float(fi);
    }
  }
  fi.close();
}

void noodle_grid_from_file(NDL_File &fi,
                           byte *buffer,
                           uint16_t K) {
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[i * K + j] = noodle_read_float(fi);
    }
  }
}

void noodle_grid_from_file(const char *fn,
                           int8_t *buffer,
                           uint16_t K) {
  fi = noodle_fs_open_read(fn);
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[i * K + j] = noodle_read_float(fi);
    }
  }
  fi.close();
}

void noodle_grid_from_file(NDL_File &fi,
                           int8_t *buffer,
                           uint16_t K) {
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[i * K + j] = noodle_read_float(fi);
    }
  }
}

void noodle_grid_from_file(const char *fn,
                           float *buffer,
                           uint16_t K) {
  fi = noodle_fs_open_read(fn);
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[i * K + j] = noodle_read_float(fi);
    }
  }
  fi.close();
}

void noodle_grid_from_file(NDL_File &fi,
                           float *buffer,
                           uint16_t K) {
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[i * K + j] = noodle_read_float(fi);
    }
  }
}
