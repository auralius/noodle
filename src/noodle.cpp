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

bool noodle_sd_init(int clk_pin,
                    int cmd_pin,
                    int d0_pin) {
#if defined(NOODLE_USE_SD_MMC)
  // SD_MMC supports setPins on some boards; begin with common args used originally
  SD_MMC.setPins(clk_pin, cmd_pin, d0_pin);
  return SD_MMC.begin("/sdcard", true, false, 20000, 5);
#else
  (void)clk_pin;
  (void)cmd_pin;
  (void)d0_pin;
  return noodle_sd_init();
#endif
}

bool noodle_sd_init() {
#if defined(NOODLE_USE_SD_MMC)
  return SD_MMC.begin("/sdcard", false, false, 20000, 5);
#else
  return NOODLE_FS.begin();
#endif
}

bool noodle_sd_init(uint8_t cs_pin) {
#if defined(NOODLE_USE_SDFAT)
  return NOODLE_FS.begin(cs_pin);
#elif defined(NOODLE_USE_SD_MMC)
  (void)cs_pin;
  return SD_MMC.begin("/sdcard", false, false, 20000, 5);
#else
  // FFat / LittleFS typically don't use CS pins
  (void)cs_pin;
  return NOODLE_FS.begin();
#endif
}

// Assumes input is between 1 and 26*26 = 676
void noodle_n2ll(uint16_t number,
                 char *out) {
  int first = number / 26;
  int second = number % 26;
  out[0] = 'a' + first;
  out[1] = 'a' + second;
}

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
  for (int16_t i = 0; i < n; i++)
    noodle_write_float(fo, array[i]);
  fo.close();
}

void noodle_array_to_file(float *array,
                          NDL_File &fo,
                          uint16_t n) {
  for (int16_t i = 0; i < n; i++)
    noodle_write_float(fo, array[i]);
}

void noodle_grid_to_file(byte *grid,
                         const char *fn,
                         uint16_t n) {
  fo = noodle_open_file_for_write(fn);
  for (int16_t i = 0; i < n; i++) {
    for (int16_t j = 0; j < n; j++) {
      noodle_write_float(fo, (float)grid[i * n + j]);
    }
  }
  fo.close();
}

void noodle_grid_to_file(byte *grid,
                         NDL_File &fo,
                         uint16_t n) {
  for (int16_t i = 0; i < n; i++) {
    for (int16_t j = 0; j < n; j++) {
      noodle_write_float(fo, (float)grid[i * n + j]);
    }
  }
}

void noodle_grid_to_file(float *grid,
                         const char *fn,
                         uint16_t n) {
  fo = noodle_open_file_for_write(fn);
  for (int16_t i = 0; i < n; i++) {
    for (int16_t j = 0; j < n; j++) {
      noodle_write_float(fo, grid[i * n + j]);
    }
  }
  fo.close();
}

void noodle_grid_to_file(float *grid,
                         NDL_File &fo,
                         uint16_t n) {
  for (int16_t i = 0; i < n; i++) {
    for (int16_t j = 0; j < n; j++) {
      noodle_write_float(fo, grid[i * n + j]);
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
  return (float)grid[(i - P) * W + (j - P)];
}

float noodle_get_padded_x(float *grid,
                          int16_t i,
                          int16_t j,
                          int16_t W,
                          int16_t P) {
  if ((i < P) || (j < P) || (i > (W - 1 + P)) || (j > (W - 1 + P))) {
    return 0.0;
  }
  return grid[(i - P) * W + (j - P)];
}

uint16_t noodle_do_bias(float *output,
                        float bias,
                        uint16_t n) {
  for (int16_t i = 0; i < n; i++) {
    for (int16_t j = 0; j < n; j++) {
      output[i * n + j] += bias;
      if (output[i * n + j] < 0.0) {
        output[i * n + j] = 0.0;
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
    const uint16_t base_y = out_y * S;
    for (uint16_t out_x = 0; out_x < Wo; out_x++) {
      const uint16_t base_x = out_x * S;

#if NOODLE_POOL_MODE == NOODLE_POOL_MAX
      float vmax = -FLT_MAX;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (base_y + win_y) * W;
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          float v = input[row + base_x + win_x];
          if (v > vmax) vmax = v;
        }
      }
      noodle_write_float(fo, vmax);

#elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
      float acc = 0.0f;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (base_y + win_y) * W;
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
    const uint16_t base_y = out_y * S;
    for (uint16_t out_x = 0; out_x < Wo; out_x++) {
      const uint16_t base_x = out_x * S;

#if NOODLE_POOL_MODE == NOODLE_POOL_MAX
      float vmax = -FLT_MAX;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (base_y + win_y) * W;
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          float v = input[row + base_x + win_x];
          if (v > vmax) vmax = v;
        }
      }
      output[out_y * Wo + out_x] = vmax;

#elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
      float acc = 0.0f;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (base_y + win_y) * W;
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          acc += input[row + base_x + win_x];
        }
      }
      output[out_y * Wo + out_x] = acc * inv_KK;
#endif
    }
  }
  return Wo;
}

uint16_t noodle_do_conv(byte *grid,
                        float *kernel,
                        uint16_t K,
                        uint16_t W,
                        float *output,
                        uint16_t P,
                        uint16_t S) {
  uint16_t V = (W - K + 2 * P) / S + 1;
  for (int16_t i = 0; i < V; i++) {
    for (int16_t j = 0; j < V; j++) {
      float v = 0;
      for (int16_t k = 0; k < K; k++) {
        for (int16_t l = 0; l < K; l++) {
          v += kernel[k * K + l] * noodle_get_padded_x(grid, i * S + k, j * S + l, W, P);
        }
      }
      output[i * V + j] += v;
    }
  }
  return V;
}

uint16_t noodle_do_conv(float *grid,
                        float *kernel,
                        uint16_t K,
                        uint16_t W,
                        float *output,
                        uint16_t P,
                        uint16_t S) {
  uint16_t V = (W - K + 2 * P) / S + 1;
  for (int16_t i = 0; i < V; i++) {
    for (int16_t j = 0; j < V; j++) {
      float v = 0;
      for (int16_t k = 0; k < K; k++) {
        for (int16_t l = 0; l < K; l++) {
          v += kernel[k * K + l] * noodle_get_padded_x(grid, i * S + k, j * S + l, W, P);
        }
      }
      output[i * V + j] += v;
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


void noodle_reset_buffer(float *buffer,
                         uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    buffer[i] = 0.0;
  }
}

uint16_t noodle_conv_byte(const char *in_fn,
                          uint16_t n_inputs,
                          uint16_t n_outputs,
                          const char *out_fn,
                          uint16_t W,
                          const Conv &conv,
                          const Pool &pool,
                          CBFPtr progress_cb) {
  byte *in_buffer = (byte *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  char i_fn[20], o_fn[20];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  float kernel[conv.K][conv.K];
  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W * W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_n2ll(I, &i_fn[4]);      
      noodle_grid_from_file(i_fn, in_buffer, W);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }
    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    noodle_n2ll(O, &o_fn[6]);
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, o_fn);
  }
  fw.close();
  fb.close();
  return V;
}

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

  char i_fn[20], o_fn[20];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  float kernel[conv.K][conv.K];
  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W * W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_n2ll(I, &i_fn[4]);      
      noodle_grid_from_file(i_fn, in_buffer, W);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }
    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    noodle_n2ll(O, &o_fn[6]);
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, o_fn);
  }
  fw.close();
  fb.close();
  return V;
}

// File → memory with float input
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  char i_fn[20];
  strcpy(i_fn, in_fn);
  float kernel[conv.K][conv.K];
  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W * W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_n2ll(I, &i_fn[4]);      
      noodle_grid_from_file(i_fn, in_buffer, W);
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

// Memory → file with float input
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer;
  float *out_buffer = (float *)temp_buff1;

  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  char o_fn[20];
  strcpy(o_fn, out_fn);
  float kernel[conv.K][conv.K];
  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W * W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {      in_buffer = noodle_slice(input, W, I);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }
    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);
    noodle_n2ll(O, &o_fn[6]);
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, o_fn);
  }
  fw.close();
  fb.close();
  return V;
}

// Memory → memory with float input
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
    for (uint16_t I = 0; I < n_inputs; I++) {      in_buffer = noodle_slice(input, W, I);
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

uint16_t noodle_flat(const char *in_fn,
                     float *output,
                     uint16_t V,
                     uint16_t n_filters) {
  char i_fn[12];
  strcpy(i_fn, in_fn);

  for (uint16_t k = 0; k < n_filters; k++) {
    noodle_n2ll(k, &i_fn[4]);
    fi = noodle_fs_open_read(i_fn);
    for (uint16_t i = 0; i < (V * V); i++) {
      output[i * n_filters + k] = noodle_read_float(fi);
    }
    fi.close();
  }
  return V * V * n_filters;
}

uint16_t noodle_flat(float *input,
                     float *output,
                     uint16_t V,
                     uint16_t n_filters) {
  for (uint16_t k = 0; k < n_filters; k++) {
    float *sliced_input = noodle_slice(input, V, k);
    for (uint16_t i = 0; i < (V * V); i++) {
      output[i * n_filters + k] = sliced_input[i];
    }
  }
  return V * V * n_filters;
}

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

uint16_t noodle_fcn(const float *input,
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
      h += input[j] * noodle_read_float(fw);
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
    fi.seek(0);
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
    fi.seek(0);
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
      float val = (idx < 0 || idx >= W) ? 0.0 : input[idx];
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

uint16_t noodle_conv1d(float *input,
                       float *output,
                       uint16_t n_inputs,
                       uint16_t n_outputs,
                       const char *in_fn,
                       const char *out_fn,
                       uint16_t W,
                       const Conv &conv,
                       const Pool &pool,
                       CBFPtr progress_cb) {
  float progress = 0;
  char i_fn[12], o_fn[12];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  float kernel[conv.K];

  uint16_t V = 0;
  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(output, W);
    float bias = noodle_read_float(fb);

    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_n2ll(I, &i_fn[4]);
      fi = noodle_fs_open_read(i_fn);
      for (uint16_t i = 0; i < W; i++) {
        input[i] = noodle_read_float(fi);
      }
      fi.close();      for (uint16_t i = 0; i < conv.K; i++) {
        kernel[i] = noodle_read_float(fw);
      }
      fw.close();

      V = noodle_do_conv1d(input, kernel, W, conv.K, output, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += 1.0 / (float)(n_inputs * n_outputs - 1);
    }

    for (uint16_t i = 0; i < V; i++) {
      output[i] += bias;
      if (conv.act == ACT_RELU && output[i] < 0.0) output[i] = 0.0;
    }

    noodle_n2ll(O, &o_fn[6]);
    V = noodle_do_pooling1d(output, V, pool.M, pool.T, o_fn);
  }

  fw.close();
  fb.close();
  return V;
}

uint16_t noodle_conv1d(float *input,
                       float *output,
                       uint16_t n_inputs,
                       uint16_t n_outputs,
                       const char *in_fn,
                       const char *out_fn,
                       uint16_t W,
                       const Conv &conv,
                       CBFPtr progress_cb) {

  float progress = 0;
  char i_fn[12], o_fn[12];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  float kernel[conv.K];
  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(output, W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_n2ll(I, &i_fn[4]);
      fi = noodle_fs_open_read(i_fn);
      for (uint16_t i = 0; i < W; i++) {
        input[i] = noodle_read_float(fi);
      }
      fi.close();      for (uint16_t i = 0; i < conv.K; i++) {
        kernel[i] = noodle_read_float(fw);
      }
      fw.close();

      V = noodle_do_conv1d(input, kernel, W, conv.K, output, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += 1.0 / (float)(n_inputs * n_outputs - 1);
    }

    for (uint16_t i = 0; i < V; i++) {
      output[i] += bias;
      if (conv.act == ACT_RELU && output[i] < 0.0) output[i] = 0.0;
    }

    noodle_n2ll(O, &o_fn[6]);
    V = noodle_do_pooling1d(output, V, 1, 1, o_fn);
  }

  fw.close();
  fb.close();
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

// Depthwise convolution: one input channel → one output channel
uint16_t noodle_dwconv_float(const char *in_fn,
                             uint16_t n_channels,
                             const char *out_fn,
                             uint16_t W,
                             const Conv &conv,     // reuse Conv: K,P,S,act, weight_fn, bias_fn
                             const Pool &pool,
                             CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0;
  float progress_step = 1.0f / (float)(n_channels > 1 ? (n_channels - 1) : 1);

  char i_fn[20], o_fn[20];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  // KxK kernel for one channel
  float kernel[conv.K][conv.K];

  // Open bias file (one bias per channel)
  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);

  uint16_t V = 0;

  for (uint16_t C = 0; C < n_channels; C++) {
    // Load one input channel plane
    noodle_n2ll(C, &i_fn[4]);              // same style as conv: channel index in filename
    noodle_grid_from_file(i_fn, in_buffer, W);

    // Load per-channel bias
    float bias = noodle_read_float(fb);

    // Load per-channel kernel:
    // We keep the same 2-slot naming as Conv2D weights, but set I==O==C    noodle_grid_from_file(fw, (float *)kernel, conv.K);

    // Compute depthwise conv for this channel
    noodle_reset_buffer(out_buffer, W * W);
    V = noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);

    // Bias + activation
    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);

    // Write pooled (or unpooled) output channel plane
    noodle_n2ll(C, &o_fn[6]);              // output channel index in filename
    V = noodle_do_pooling(out_buffer, V, pool.M, pool.T, o_fn);

    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fw.close();
  fb.close();
  return V;
}

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
  float kernel[conv.K][conv.K];

  // Streamed per-layer weight+bias files
  NDL_File fb = noodle_fs_open_read(conv.bias_fn);
  NDL_File fw = noodle_fs_open_read(conv.weight_fn);

  uint16_t V = 0;

  // Exporter layout for DW (M==1 case):
  // - weights: for C in [0..Cin): write K*K floats (row-major)
  // - bias   : for C in [0..Cin): write 1 float
  //
  // This function assumes multiplier M == 1.
  for (uint16_t C = 0; C < n_channels; C++) {
    float *in_plane = noodle_slice(input, W, C);

    // Bias: one per channel
    const float bias = noodle_read_float(fb);

    // Kernel: next K*K floats from stream
    noodle_grid_from_file(fw, (float *)kernel, conv.K);

    // Convolution into scratch buffer
    noodle_reset_buffer(out_buffer, (uint16_t)(W * W));
    V = noodle_do_conv(in_plane,
                       (float *)kernel,
                       conv.K,
                       W,
                       out_buffer,
                       conv.P,
                       conv.S);

    // Bias + activation
    V = noodle_do_bias_act(out_buffer, bias, V, conv.act);

    // Pooling into output plane
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



void noodle_gap(float *x_chw,
                uint16_t C,
                uint16_t W) {
  const uint32_t n = (uint32_t)W * (uint32_t)W;
  for (uint16_t c = 0; c < C; c++) {
    float *plane = x_chw + (uint32_t)c * n;
    double acc = 0.0;
    for (uint32_t i = 0; i < n; i++) acc += (double)plane[i];
    // Write to the front of the buffer. Safe because we finish reading the plane first.
    x_chw[c] = (float)(acc / (double)n);
  }
}
