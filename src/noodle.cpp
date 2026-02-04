#include "noodle.h"
#include <math.h>
#include <float.h>
#include <stdint.h>

#if defined(NOODLE_USE_SDFAT)
SdFat NOODLE_FS;  // define the SdFat object declared in noodle_fs.h
#endif

// File handles and temp buffers (backend-agnostic)
NDL_File fw, fb, fo, fi;
static void *temp_buff1 = NULL;
static void *temp_buff2 = NULL;


inline float *noodle_slice(float *flat,
                                  size_t W,
                                  size_t z) {
  return flat + z * W * W;
}

void noodle_setup_temp_buffers(void *b1,
                               void *b2) {
  temp_buff1 = b1;
  temp_buff2 = b2;
}

NDL_File noodle_open_file_for_write(const char *fn) {
  noodle_fs_remove(fn);
  return noodle_fs_open_write(fn);
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
  return NOODLE_FS.begin(cs_pin);

#elif defined(NOODLE_USE_SD_MMC)
  (void)cs_pin;
  return SD_MMC.begin("/sdcard", false, false, 20000, 5);

#else
  // FFat / LittleFS ignore CS
  (void)cs_pin;
  return NOODLE_FS.begin();
#endif
}

#if defined(NOODLE_USE_SD_MMC)
  #include "esp_vfs_fat.h"
  #include "sdmmc_cmd.h"
  #include "driver/sdmmc_host.h"
#endif


float noodle_read_float(NDL_File &f) {
  char s[20];
  size_t n = noodle_read_bytes_until(f, '\n', (char *)s, sizeof(s));
  s[n] = '\0';
  return atof(s);
}

byte noodle_read_byte(NDL_File &f) {
  char s[20];
  size_t n = noodle_read_bytes_until(f, '\n', (char *)s, sizeof(s));
  s[n] = '\0';
  return (byte)atoi(s);
}

void noodle_write_float(NDL_File &f,
                        float d) {
  f.println(d, 6);
}

void noodle_write_byte(NDL_File &f,
                       byte d) {
  f.println(d);
}

void noodle_delete_file(const char *fn) {
  noodle_fs_remove(fn);
}

float *noodle_create_buffer(uint16_t size) {
  return (float *)malloc(size);
}

void noodle_delete_buffer(float *buffer) {
  free(buffer);
}

void noodle_array_to_file(float *array,
                          const char *fn,
                          uint16_t n) {
  fo = noodle_open_file_for_write(fn);
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
  fo = noodle_open_file_for_write(fn);
  for (uint16_t i = 0; i < n; i++) {
    const uint32_t row = (uint32_t)i * (uint32_t)n;
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
    const uint32_t row = (uint32_t)i * (uint32_t)n;
    for (uint16_t j = 0; j < n; j++) {
      noodle_write_float(fo, (float)grid[row + j]);
    }
  }
}

void noodle_grid_to_file(float *grid,
                         const char *fn,
                         uint16_t n) {
  fo = noodle_open_file_for_write(fn);
  for (uint16_t i = 0; i < n; i++) {
    const uint32_t row = (uint32_t)i * (uint32_t)n;
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
    const uint32_t row = (uint32_t)i * (uint32_t)n;
    for (uint16_t j = 0; j < n; j++) {
      noodle_write_float(fo, grid[row + j]);
    }
  }
}

float noodle_get_padded_x(byte *grid,
                          int16_t i,
                          int16_t j,
                          int16_t W,
                          int16_t P) {
  if ((i < P) || (j < P) || (i > (W - 1 + P)) || (j > (W - 1 + P))) {
    return 0.0;
  }
  return (float)grid[(int32_t)(i - P) * (int32_t)W + (int32_t)(j - P)];
}

float noodle_get_padded_x(float *grid,
                          int16_t i,
                          int16_t j,
                          int16_t W,
                          int16_t P) {
  if ((i < P) || (j < P) || (i > (W - 1 + P)) || (j > (W - 1 + P))) {
    return 0.0;
  }
  return grid[(int32_t)(i - P) * (int32_t)W + (int32_t)(j - P)];
}

uint16_t noodle_do_bias(float *output,
                        float bias,
                        uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    const uint32_t row = (uint32_t)i * (uint32_t)n;
    for (uint16_t j = 0; j < n; j++) {
      const uint32_t idx = row + j;
      output[idx] += bias;
      if (output[idx] < 0.0f) {
        output[idx] = 0.0f;
      }
    }
  }
  return n;
}

uint16_t noodle_do_pooling(float *input,
                           uint16_t W,
                           uint16_t K,
                           uint16_t S,
                           const char *fn) {
  const uint16_t Wo = (W - K) / S + 1;
  fo = noodle_open_file_for_write(fn);

#if NOODLE_POOL_MODE == NOODLE_POOL_MEAN
  const float inv_KK = 1.0f / (float)(K * K);
#endif

  for (uint16_t out_y = 0; out_y < Wo; out_y++) {
    const uint32_t base_y = (uint32_t)out_y * (uint32_t)S;
    for (uint16_t out_x = 0; out_x < Wo; out_x++) {
      const uint32_t base_x = (uint32_t)out_x * (uint32_t)S;

#if NOODLE_POOL_MODE == NOODLE_POOL_MAX
      float vmax = -FLT_MAX;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint32_t row = ((uint32_t)base_y + (uint32_t)win_y) * (uint32_t)W;
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          float v = input[row + base_x + win_x];
          if (v > vmax) vmax = v;
        }
      }
      noodle_write_float(fo, vmax);

#elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
      float acc = 0.0f;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint32_t row = ((uint32_t)base_y + (uint32_t)win_y) * (uint32_t)W;
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          acc += input[row + base_x + win_x];
        }
      }
      noodle_write_float(fo, acc * inv_KK);
#endif
    }
  }

  fo.close();
  return Wo;
}

uint16_t noodle_do_pooling(float *input,
                           uint16_t W,
                           uint16_t K,
                           uint16_t S,
                           NDL_File &fo) {
  const uint16_t Wo = (W - K) / S + 1;
  
#if NOODLE_POOL_MODE == NOODLE_POOL_MEAN
  const float inv_KK = 1.0f / (float)(K * K);
#endif

  for (uint16_t out_y = 0; out_y < Wo; out_y++) {
    const uint32_t base_y = (uint32_t)out_y * (uint32_t)S;
    for (uint16_t out_x = 0; out_x < Wo; out_x++) {
      const uint32_t base_x = (uint32_t)out_x * (uint32_t)S;

#if NOODLE_POOL_MODE == NOODLE_POOL_MAX
      float vmax = -FLT_MAX;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint32_t row = ((uint32_t)base_y + (uint32_t)win_y) * (uint32_t)W;
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          float v = input[row + base_x + win_x];
          if (v > vmax) vmax = v;
        }
      }
      noodle_write_float(fo, vmax);

#elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
      float acc = 0.0f;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint32_t row = ((uint32_t)base_y + (uint32_t)win_y) * (uint32_t)W;
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          acc += input[row + base_x + win_x];
        }
      }
      noodle_write_float(fo, acc * inv_KK);
#endif
    }
  }

  return Wo;
}

uint16_t noodle_do_pooling(const float *input,
                           uint16_t W,
                           uint16_t K,
                           uint16_t S,
                           float *output) {
  const uint16_t Wo = (W - K) / S + 1;

#if NOODLE_POOL_MODE == NOODLE_POOL_MEAN
  const float inv_KK = 1.0f / (float)(K * K);
#endif

  for (uint16_t out_y = 0; out_y < Wo; out_y++) {
    const uint32_t base_y = (uint32_t)out_y * (uint32_t)S;
    for (uint16_t out_x = 0; out_x < Wo; out_x++) {
      const uint32_t base_x = (uint32_t)out_x * (uint32_t)S;

#if NOODLE_POOL_MODE == NOODLE_POOL_MAX
      float vmax = -FLT_MAX;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint32_t row = ((uint32_t)base_y + (uint32_t)win_y) * (uint32_t)W;
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          float v = input[row + base_x + win_x];
          if (v > vmax) vmax = v;
        }
      }
      output[(uint32_t)out_y * (uint32_t)Wo + out_x] = vmax;

#elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
      float acc = 0.0f;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint32_t row = ((uint32_t)base_y + (uint32_t)win_y) * (uint32_t)W;
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          acc += input[row + base_x + win_x];
        }
      }
      output[(uint32_t)out_y * (uint32_t)Wo + out_x] = acc * inv_KK;
#endif
    }
  }
  return Wo;
}

uint16_t noodle_do_conv(byte *grid,
                        const float *kernel,
                        uint16_t K,
                        uint16_t W,
                        float *output,
                        uint16_t P,
                        uint16_t S) {
  uint16_t V = (W - K + 2 * P) / S + 1;
  for (uint16_t i = 0; i < V; i++) {
    for (uint16_t j = 0; j < V; j++) {
      float v = 0;
      for (uint16_t k = 0; k < K; k++) {
        for (uint16_t l = 0; l < K; l++) {
          v += kernel[k * K + l] * noodle_get_padded_x(grid, i * S + k, j * S + l, W, P);
        }
      }
      output[(uint32_t)i * (uint32_t)V + (uint32_t)j] += v;
    }
  }
  return V;
}

uint16_t noodle_do_conv(float *grid,
                        const float *kernel,
                        uint16_t K,
                        uint16_t W,
                        float *output,
                        uint16_t P,
                        uint16_t S) {
  uint16_t V = (W - K + 2 * P) / S + 1;
  for (uint16_t i = 0; i < V; i++) {
    for (uint16_t j = 0; j < V; j++) {
      float v = 0;
      for (uint16_t k = 0; k < K; k++) {
        for (uint16_t l = 0; l < K; l++) {
          v += kernel[k * K + l] * noodle_get_padded_x(grid, i * S + k, j * S + l, W, P);
        }
      }
      output[(uint32_t)i * (uint32_t)V + (uint32_t)j] += v;
    }
  }
  return V;
}

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
      buffer[(uint32_t)i * (uint32_t)K + j] = noodle_read_float(fi);
    }
  }
  fi.close();
}

void noodle_grid_from_file(NDL_File &fi,
                           byte *buffer,
                           uint16_t K) {
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[(uint32_t)i * (uint32_t)K + j] = noodle_read_float(fi);
    }
  }
}

void noodle_grid_from_file(const char *fn,
                           int8_t *buffer,
                           uint16_t K) {
  fi = noodle_fs_open_read(fn);
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[(uint32_t)i * (uint32_t)K + j] = noodle_read_float(fi);
    }
  }
  fi.close();
}

void noodle_grid_from_file(NDL_File &fi,
                           int8_t *buffer,
                           uint16_t K) {
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[(uint32_t)i * (uint32_t)K + j] = noodle_read_float(fi);
    }
  }
}

void noodle_grid_from_file(const char *fn,
                           float *buffer,
                           uint16_t K) {
  fi = noodle_fs_open_read(fn);
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[(uint32_t)i * (uint32_t)K + j] = noodle_read_float(fi);
    }
  }
  fi.close();
}

void noodle_grid_from_file(NDL_File &fi,
                           float *buffer,
                           uint16_t K) {
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[(uint32_t)i * (uint32_t)K + j] = noodle_read_float(fi);
    }
  }
}


void noodle_reset_buffer(float *buffer,
                         uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    buffer[i] = 0.0;
  }
}

// File → file with byte input and Conv
uint16_t noodle_conv_byte(const char *in_fn,
                          uint16_t n_inputs,
                          uint16_t n_outputs,
                          const char *out_fn,
                          uint16_t W,
                          const Conv &conv,
                          const Pool &pool,
                          CBFPtr progress_cb) {
  byte  *in_buffer  = (byte *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0.0f;
  const uint32_t total = (uint32_t)n_inputs * (uint32_t)n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);      
  fo = noodle_fs_open_write(out_fn);      

  uint16_t V = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K]; 

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W * W);
    float bias = noodle_read_float(fb);
    noodle_rewind_file(fi); // rewind input file for each output channel
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_grid_from_file(fi, in_buffer, W);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, fo);
  }

  fw.close(); 
  fb.close(); 
  fi.close(); 
  fo.close();
  return V;
}


// File → file with float input and Conv
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);
  fo = noodle_fs_open_write(out_fn);

  uint16_t V = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];
  
  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W * W);
    float bias = noodle_read_float(fb);
    noodle_rewind_file(fi); // rewind input file for each output channel
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_grid_from_file(fi, in_buffer, W);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, fo);
  }
  fw.close();
  fb.close();
  fi.close();
  fo.close();
  return V;
}

// File → memory with float input with Conv
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0.0f;
  const uint32_t total = (uint32_t)n_inputs * (uint32_t)n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);  // packed input

  uint16_t V = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K]; 

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W * W);
    float bias = noodle_read_float(fb);

    // rewind packed input for each output filter
    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_grid_from_file(fi, in_buffer, W);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);

    const uint16_t Wo = (V - pool.M) / pool.T + 1;
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, noodle_slice(output, Wo, O));
  }

  fw.close(); 
  fb.close(); 
  fi.close();
  return V;
}


// Memory → file with float input and Conv
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *out_buffer = (float *)temp_buff2;

  float progress = 0.0f;
  const uint32_t total = (uint32_t)n_inputs * (uint32_t)n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fo = noodle_fs_open_write(out_fn);   // packed output

  uint16_t V = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, (uint16_t)((uint32_t)W * (uint32_t)W));
    float bias = noodle_read_float(fb);

    for (uint16_t I = 0; I < n_inputs; I++) {
      float *in_buffer = noodle_slice(input, W, I);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);

      V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, fo);
  }

  fw.close(); 
  fb.close(); 
  fo.close();
  return V;
}

// Memory → file with float input and ConvMem
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const ConvMem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb)
{
  float *out_buffer = (float *)temp_buff1;

  float progress = 0.0f;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  fo = noodle_fs_open_write(out_fn);

  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, (uint16_t)((uint32_t)W * (uint32_t)W));
    const float bias = conv.bias[O];

    // Accumulate over input channels
    for (uint16_t I = 0; I < n_inputs; I++) {
      const float *kernel = conv.weight + (uint32_t)(O * n_inputs + I) * (uint32_t)(conv.K * conv.K);
      float *in_plane = noodle_slice(input, W, I);  // expects CHW in memory
      V = noodle_do_conv(in_plane, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, fo);
  }

  fo.close();
  return V;  // spatial size after pooling
}

// Memory → memory with float input and Conv
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer;
  float *out_buffer = (float *)temp_buff1;

  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);
  float kernel[conv.K][conv.K];
  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W * W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {      
      in_buffer = noodle_slice(input, W, I);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }
    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    uint16_t Wo = (V - pool.M) / pool.T + 1;
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, noodle_slice(output, Wo, O));
  }
  fw.close();
  fb.close();
  return V;
}

// Memory → memory with float input and ConvMem
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const ConvMem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer;
  float *out_buffer = (float *)temp_buff1;

  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);
  
  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W * W);
    float bias = conv.bias[O];
    for (uint16_t I = 0; I < n_inputs; I++) {      
      const float *kernel = conv.weight + (O * n_inputs + I) * (conv.K * conv.K);
      in_buffer = noodle_slice(input, W, I);
      V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }
    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    uint16_t Wo = (V - pool.M) / pool.T + 1;
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, noodle_slice(output, Wo, O));
  }
  return V;
}

// File CHW → Memory HWC-flatten
uint16_t noodle_flat(const char *in_fn,
                     float *output,
                     uint16_t V,
                     uint16_t n_filters) {
  fi = noodle_fs_open_read(in_fn);

  const uint32_t plane = (uint32_t)V * (uint32_t)V;

  // Input file is CHW: [ch0 plane][ch1 plane]...[chK plane]
  // Output wants HWC-flatten: output[i*n_filters + k]
  for (uint16_t k = 0; k < n_filters; k++) {
    for (uint32_t i = 0; i < plane; i++) {
      float x = noodle_read_float(fi);
      output[i * (uint32_t)n_filters + k] = x;
    }
  }

  fi.close();
  // NOTE: return type is uint16_t for compatibility; caller should ensure plane*n_filters <= 65535 if it matters.
  return (uint16_t)(plane * (uint32_t)n_filters);
}


// Memory CHW → Memory HWC-flatten
uint16_t noodle_flat(float *input,
                     float *output,
                     uint16_t V,
                     uint16_t n_filters) {
  const uint32_t plane = (uint32_t)V * (uint32_t)V;
  for (uint16_t k = 0; k < n_filters; k++) {
    float *sliced_input = noodle_slice(input, V, k);
    for (uint32_t i = 0; i < plane; i++) {
      output[i * (uint32_t)n_filters + k] = sliced_input[i];
    }
  }
  return (uint16_t)(plane * (uint32_t)n_filters);
}

// Memory HWC-flatten → File HWC-flatten
uint16_t noodle_fcn(const int8_t *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0.0;
  float progress_step = 1.0f / (float)(n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fo = noodle_open_file_for_write(out_fn);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      h += (float)input[j] * noodle_read_float(fw);
    if ((h < 0.0) && (fcn.act == ACT_RELU)) h = 0.0;
    noodle_write_float(fo, h);
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

// Memory HWC-flatten → File HWC-flatten
uint16_t noodle_fcn(const byte *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fo = noodle_open_file_for_write(out_fn);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      h += (float)input[j] * noodle_read_float(fw);
    if ((h < 0.0) && (fcn.act == ACT_RELU)) h = 0.0;
    noodle_write_float(fo, h);
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

// Memory HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const byte *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);

  for (uint16_t k = 0; k < n_outputs; k++) {
    output[k] = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      output[k] += (float)input[j] * noodle_read_float(fw);
    if ((fcn.act == ACT_RELU) && (output[k] < 0.f)) output[k] = 0.0f;
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fw.close();
  fb.close();

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);

  return n_outputs;
}

// Memory HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const float *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);

  for (uint16_t k = 0; k < n_outputs; k++) {
    output[k] = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      output[k] += input[j] * noodle_read_float(fw);
    if ((fcn.act == ACT_RELU) && (output[k] < 0.f)) output[k] = 0.f;

    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fw.close();
  fb.close();

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);

  return n_outputs;
}

// Memory HWC-flatten → File (streamed outputs)
uint16_t noodle_fcn(const float *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb)
{
  float progress = 0.f;
  float progress_step = (n_outputs > 1) ? (1.0f / (float)(n_outputs - 1)) : 1.0f;

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fo = noodle_open_file_for_write(out_fn);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float output = noodle_read_float(fb);

    for (uint16_t j = 0; j < n_inputs; j++) {
      output += input[j] * noodle_read_float(fw);
    }

    if ((fcn.act == ACT_RELU) && (output < 0.f)) output = 0.f;

    noodle_write_float(fo, output);

    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fw.close();
  fb.close();
  fo.close();
  return n_outputs;
}

// File HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fi = noodle_fs_open_read(in_fn);

  for (uint16_t j = 0; j < n_outputs; j++) {
    output[j] = noodle_read_float(fb);
    noodle_rewind_file(fi);
    for (uint16_t k = 0; k < n_inputs; k++)
      output[j] += noodle_read_float(fi) * noodle_read_float(fw);
    if ((output[j] < 0.0) && (fcn.act == ACT_RELU)) output[j] = 0.0;
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fi.close();
  fw.close();
  fb.close();

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);

  return n_outputs;
}

// Memory HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const int8_t *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);

  for (uint16_t j = 0; j < n_outputs; j++) {
    output[j] = noodle_read_float(fb);
    for (uint16_t k = 0; k < n_inputs; k++)
      output[j] += (float)input[k] * noodle_read_float(fw);
    if ((output[j] < 0.0) && (fcn.act == ACT_RELU)) output[j] = 0.0;
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fw.close();
  fb.close();

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);

  return n_outputs;
}

// File HWC-flatten → File HWC-flatten
uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fo = noodle_open_file_for_write(out_fn);
  fi = noodle_fs_open_read(in_fn);

  for (uint16_t j = 0; j < n_outputs; j++) {
    float h = noodle_read_float(fb);
    noodle_rewind_file(fi);
    for (uint16_t k = 0; k < n_inputs; k++)
      h += noodle_read_float(fi) * noodle_read_float(fw);
    if ((h < 0.0) && (fcn.act == ACT_RELU)) h = 0.0;
    noodle_write_float(fo, h);
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fi.close();
  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

// Memory HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const float *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNMem &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_outputs - 1);

  uint16_t l = 0;
  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = fcn.bias[k];
    for (uint16_t j = 0; j < n_inputs; j++)
      h += input[j] * fcn.weight[l++];  // <-- was ++l
    if ((fcn.act == ACT_RELU) && (h < 0.f)) h = 0.f;
    output[k] = h;
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);
  
  return n_outputs;
}

uint16_t noodle_soft_max(float *input_output,
                         uint16_t n) {
  float max_val = input_output[0];
  for (int i = 1; i < (int)n; i++) {
    if (input_output[i] > max_val)
      max_val = input_output[i];
  }

  float sum = 0.0;
  for (int i = 0; i < (int)n; i++) {
    input_output[i] = expf(input_output[i] - max_val);
    sum += input_output[i];
  }

  for (int i = 0; i < (int)n; i++) {
    input_output[i] /= sum;
  }
  return n;
}

uint16_t noodle_sigmoid(float *input_output,
                        uint16_t n) {
  for (int i = 0; i < (int)n; i++) {
    input_output[i] = 1.0f / (1.0f + expf(-input_output[i]));
  }
  return n;
}

uint16_t noodle_relu(float *input_output,
                        uint16_t n) {
  for (int i = 0; i < (int)n; i++) {
    input_output[i] = input_output[i] > 0.0f ? input_output[i] : 0.0f;
  }
  return n;
}

uint16_t noodle_do_conv1d(float *input,
                          float *kernel,
                          uint16_t W,
                          uint16_t K,
                          float *output,
                          uint16_t P,
                          uint16_t S) {
  uint16_t V = (W - K + 2 * P) / S + 1;

  for (uint16_t i = 0; i < V; i++) {
    float acc = 0.0;
    for (uint16_t k = 0; k < K; k++) {
      int16_t idx = i * S + k - P;
      float val = (idx < 0 || idx >= (int16_t)W) ? 0.0 : input[idx];
      acc += val * kernel[k];
    }
    output[i] += acc;
  }

  return V;
}

uint16_t noodle_do_pooling1d(float *input,
                             uint16_t W,
                             uint16_t K,
                             uint16_t S,
                             const char *fn) {
  fo = noodle_open_file_for_write(fn);

  uint16_t Wo = (W - K) / S + 1;
  for (uint16_t i = 0; i < Wo; i++) {
    float v = -FLT_MAX;
    const uint16_t base = (uint16_t)(i * S);
    for (uint16_t k = 0; k < K; k++) {
      if (v < input[base + k]) v = input[base + k];
    }
    noodle_write_float(fo, v);
  }

  fo.close();
  return Wo;
}

uint16_t noodle_do_pooling1d(float *input,
                             uint16_t W,
                             uint16_t K,
                             uint16_t S,
                             NDL_File &fo) {
  uint16_t Wo = (W - K) / S + 1;
  for (uint16_t i = 0; i < Wo; i++) {
    float v = -FLT_MAX;
    const uint16_t base = (uint16_t)(i * S);
    for (uint16_t k = 0; k < K; k++) {
      if (v < input[base + k]) v = input[base + k];
    }
    noodle_write_float(fo, v);
  }
  return Wo;
}

uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const Conv &conv,
                       const Pool &pool,
                       CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0.0f;
  const uint32_t total = (uint32_t)n_inputs * (uint32_t)n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);      // packed input CHW
  fo = noodle_fs_open_write(out_fn);    // packed output CHW

  uint16_t V = 0;
  float kernel[NOODLE_MAX_K];

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W);
    const float bias = noodle_read_float(fb);

    // rewind packed input for each output channel
    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      for (uint16_t i = 0; i < W; i++) {
        in_buffer[i] = noodle_read_float(fi);
      }
      for (uint16_t k = 0; k < conv.K; k++) {
        kernel[k] = noodle_read_float(fw);
      }

      V = noodle_do_conv1d(in_buffer, kernel, W, conv.K, out_buffer, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    // Bias + activation in-place
    for (uint16_t i = 0; i < V; i++) {
      float v = out_buffer[i] + bias;
      if ((conv.act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      out_buffer[i] = v;
    }

    // Pool and append to packed output file
    V = noodle_do_pooling1d(out_buffer, V, pool.M, pool.T, fo);
  }

  fw.close();
  fb.close();
  fi.close();
  fo.close();
  return V;
}

uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const Conv &conv,
                       CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0.0f;
  const uint32_t total = (uint32_t)n_inputs * (uint32_t)n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);      // packed input CHW
  fo = noodle_fs_open_write(out_fn);    // packed output CHW

  uint16_t V = 0;
  float kernel[NOODLE_MAX_K];

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W);
    const float bias = noodle_read_float(fb);

    // rewind packed input for each output channel
    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      for (uint16_t i = 0; i < W; i++) {
        in_buffer[i] = noodle_read_float(fi);
      }
      for (uint16_t k = 0; k < conv.K; k++) {
        kernel[k] = noodle_read_float(fw);
      }

      V = noodle_do_conv1d(in_buffer, kernel, W, conv.K, out_buffer, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    for (uint16_t i = 0; i < V; i++) {
      float v = out_buffer[i] + bias;
      if ((conv.act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      out_buffer[i] = v;
    }

    // No pooling: append raw V samples for this output channel.
    for (uint16_t i = 0; i < V; i++) {
      noodle_write_float(fo, out_buffer[i]);
    }
  }

  fw.close();
  fb.close();
  fi.close();
  fo.close();
  return V;
}

// Memory -> Memory Conv1D
uint16_t noodle_conv1d(const float *in,
                       uint16_t n_inputs,
                       float *out,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1; // needs >= W floats
  float *out_buffer = (float *)temp_buff2; // needs >= W floats (safe upper bound)

  float progress = 0.0f;
  const uint32_t total = (uint32_t)n_inputs * (uint32_t)n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W);
    const float bias = conv.bias[O];

    for (uint16_t I = 0; I < n_inputs; I++) {
      // read W samples of input channel I into in_buffer (same as your fi reads)
      const float *in_ch = in + (uint32_t)I * (uint32_t)W;
      for (uint16_t i = 0; i < W; i++) {
        in_buffer[i] = in_ch[i];
      }

      const float *kernel = conv.weight
        + ((uint32_t)O * (uint32_t)n_inputs + (uint32_t)I) * (uint32_t)conv.K;

      // Accumulate into out_buffer
      V = noodle_do_conv1d(in_buffer,
                           (float*)kernel,      // noodle_do_conv1d expects float*
                           W,
                           conv.K,
                           out_buffer,
                           conv.P,
                           conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    // Bias + activation in-place (same as your code)
    for (uint16_t i = 0; i < V; i++) {
      float v = out_buffer[i] + bias;
      if ((conv.act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      out_buffer[i] = v;
    }

    // "append raw V samples for this output channel" -> write to out plane
    // packed output CHW: [O0 plane][O1 plane]...
    float *out_ch = out + (uint32_t)O * (uint32_t)V;
    for (uint16_t i = 0; i < V; i++) {
      out_ch[i] = out_buffer[i];
    }
  }

  return V;
}


uint16_t noodle_do_bias_act(float *output,
                            float bias,
                            uint16_t n,
                            Activation act) {
  for (uint16_t i = 0; i < n; ++i) {
    float *row = output + i * n;
    for (uint16_t j = 0; j < n; ++j) {
      float v = row[j] + bias;
      if ((act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      row[j] = v;
    }
  }
  return n;
}

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

// File CHW → File CHW with Conv
uint16_t noodle_dwconv_float(const char *in_fn,
                             uint16_t n_channels,
                             const char *out_fn,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb)
{
  float *in_buffer  = (float *)temp_buff1;  // holds one W*W plane
  float *out_buffer = (float *)temp_buff2;  // holds one V*V plane (<= W*W)

  float progress = 0.0f;
  float progress_step = 1.0f / (float)(n_channels > 1 ? (n_channels - 1) : 1);

  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  fi = noodle_fs_open_read(in_fn);
  fb = noodle_fs_open_read(conv.bias_fn);     // one bias per channel
  fw = noodle_fs_open_read(conv.weight_fn);   // one KxK kernel per channel (stacked)
  fo = noodle_fs_open_write(out_fn);

  uint16_t V = 0;

  for (uint16_t C = 0; C < n_channels; C++) {
    noodle_grid_from_file(fi, in_buffer, W);
    float bias = noodle_read_float(fb);
    noodle_grid_from_file(fw, (float *)kernel, conv.K);

    noodle_reset_buffer(out_buffer, W * W);
    V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);

    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, fo);

    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fi.close();
  fw.close();
  fb.close();
  fo.close();

  return V;
}

// Memory CHW → Memory CHW with float input and Conv
uint16_t noodle_dwconv_float(float *input,
                             uint16_t n_channels,
                             float *output,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb)
{
  // Scratch buffer for intermediate conv output (size W*W floats)
  float *out_buffer = (float *)temp_buff1;

  float progress = 0.0f;
  const float denom = (float)((n_channels > 1) ? (n_channels - 1) : 1);
  const float progress_step = (denom > 0.0f) ? (1.0f / denom) : 1.0f;

  // Kernel buffer (K*K floats)
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  // Streamed per-layer weight+bias files
  NDL_File fb = noodle_fs_open_read(conv.bias_fn);
  NDL_File fw = noodle_fs_open_read(conv.weight_fn);

  uint16_t V = 0;

  // This function assumes multiplier M == 1.
  for (uint16_t C = 0; C < n_channels; C++) {
    float *in_plane = noodle_slice(input, W, C);
    const float bias = noodle_read_float(fb);
    noodle_grid_from_file(fw, (float *)kernel, conv.K);

    noodle_reset_buffer(out_buffer, (uint16_t)(W * W));
    V = noodle_do_conv(in_plane,
                       (float *)kernel,
                       conv.K,
                       W,
                       out_buffer,
                       conv.P,
                       conv.S);

    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    const uint16_t Wo = (uint16_t)((V - pool.M) / pool.T + 1);
    float *out_plane = noodle_slice(output, Wo, C);
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, out_plane);

    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fw.close();
  fb.close();
  return V; // spatial size after pooling
}

void noodle_gap(float *inout,
                uint16_t C,
                uint16_t W) {
  const uint32_t n = (uint32_t)W * (uint32_t)W;
  for (uint16_t c = 0; c < C; c++) {
    float *plane = inout + (uint32_t)c * n;
    double acc = 0.0;
    for (uint32_t i = 0; i < n; i++) acc += (double)plane[i];
    // Write to the front of the buffer. Safe because we finish reading the plane first.
    inout[c] = (float)(acc / (double)n);
  }
}

void noodle_bn(float *x,
               uint16_t C,
               uint16_t W,
               const float *gamma,
               const float *beta,
               const float *mean,
               const float *var,
               float eps) {
  const uint32_t plane = (uint32_t)W * (uint32_t)W;
  for (uint16_t c = 0; c < C; ++c) {
    const float inv_std = 1.0f / sqrtf(var[c] + eps);
    const float s = gamma[c] * inv_std;
    const float t = beta[c] - s * mean[c];

    float *p = x + (uint32_t)c * plane;
    for (uint32_t i = 0; i < plane; ++i) p[i] = s * p[i] + t;
  }
}


void noodle_find_max(float *input,
                     uint16_t n,
                     float &max_val,
                     uint16_t &max_idx) {
  max_val = input[0];
  max_idx = 0;
  for (uint16_t i = 1; i < n; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
      max_idx = i;
    }
  }
}