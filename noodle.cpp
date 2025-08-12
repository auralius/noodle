/**
 * @SdFile noodle.cpp
 * @brief Implementations of noodle.h.
 *
 * @authors
 * - Auralius Manurung -- Universitas Telkom, Bandung (auralius.manurung@ieee.org)
 * - Lisa Kristiana -- ITENAS, Bandung (lisa@itenas.ac.id)
 *
 * @license MIT License
 */

#include "noodle.h"

SdFat SD;

size_t noodle_read_bytes_until(SdFile &file, char terminator, char *buffer, size_t length) {
  size_t count = 0;
  int c;

  while (count < length - 1) {  // leave space for null terminator
    c = file.read();
    if (c < 0 || (char)c == terminator) break;
    buffer[count++] = (char)c;
  }

  buffer[count] = '\0';  // null-terminate
  return count;
}

bool noodle_sd_init(uint8_t cs_pin) {
  return SD.begin(cs_pin, SD_SCK_MHZ(8));
}

// Assumes input is between 1 and 26*26 = 676
void noodle_n2ll(uint16_t number, char *out) {
  int first = number / 26;
  int second = number % 26;

  out[0] = 'a' + first;
  out[1] = 'a' + second;
}

float noodle_read_float(SdFile &f) {
  char s[20];
  size_t n = noodle_read_bytes_until(f, '\n', (char *)s, sizeof(s));
  s[n] = '\0';
  return atof(s);
}

byte noodle_read_byte(SdFile &f) {
  char s[20];
  size_t n = noodle_read_bytes_until(f, '\n', (char *)s, sizeof(s));
  s[n] = '\0';
  return (byte)atoi(s);
}

void noodle_write_float(SdFile &f, float d) {
  f.println(d, 6);
}

void noodle_write_byte(SdFile &f, byte d) {
  f.println(d);
}

void noodle_delete_file(const char *fn) {
  SD.remove(fn);
}

float *noodle_create_buffer(uint16_t size) {
  return (float *)malloc(size);
}

void noodle_delete_buffer(float *buffer) {
  free(buffer);
}

void noodle_array_to_file(float *array, const char *fn, uint16_t n) {
  SdFile fo;
  fo.open(fn, O_WRITE | O_CREAT | O_TRUNC);
  for (int16_t i = 0; i < n; i++)
    noodle_write_float(fo, array[i]);
  fo.close();
}

void noodle_grid_to_file(byte *grid, const char *fn, uint16_t n) {
  SdFile fo;
  fo.open(fn, O_WRITE | O_CREAT | O_TRUNC);
  for (int16_t i = 0; i < n; i++) {
    for (int16_t j = 0; j < n; j++) {
      noodle_write_float(fo, (float)grid[i * n + j]);
    }
  }
  fo.close();
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

uint16_t noodle_do_bias(float *output, float bias, uint16_t n) {
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

uint16_t noodle_do_pooling(float *buffer,
                           uint16_t W,
                           uint16_t K,
                           uint16_t S,
                           const char *fn) {
  uint16_t Wo = (W - K) / S + 1;
  SdFile fo;
  fo.open(fn, O_WRITE | O_CREAT | O_TRUNC);

  for (int16_t i = 0; i < Wo; i++) {
    for (int16_t j = 0; j < Wo; j++) {
      float v = 0;
      for (int16_t k = 0; k < K; k++) {
        for (int16_t l = 0; l < K; l++) {
          if (v < buffer[(i * S) * W + (j * S)])
            v = buffer[(i * S) * W + (j * S)];
        }
      }
      noodle_write_float(fo, v);
    }
  }

  fo.close();
  return Wo;
}

uint16_t noodle_do_conv(byte *grid,
                        float *kernel,
                        uint16_t K,
                        uint16_t W,
                        float *output_buffer,
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
      output_buffer[i * V + j] += v;
    }
  }
  return V;
}

void noodle_array_from_file(const char *fn,
                            float *buffer,
                            uint16_t K) {
  SdFile fi;
  fi.open(fn, O_READ);
  for (uint16_t i = 0; i < K; i++)
    buffer[i] = noodle_read_float(fi);
  fi.close();
}

void noodle_grid_from_file(const char *fn,
                           byte *buffer,
                           uint16_t K,
                           bool transposed) {
  SdFile fi;
  fi.open(fn, O_READ);
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[transposed ? j * K + i : i * K + j] = noodle_read_float(fi);
    }
  }
  fi.close();
}

void noodle_grid_from_file(const char *fn,
                           int8_t *buffer,
                           uint16_t K,
                           bool transposed) {
  SdFile fi;
  fi.open(fn, O_READ);
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[transposed ? j * K + i : i * K + j] = noodle_read_float(fi);
    }
  }
  fi.close();
}

void noodle_grid_from_file(const char *fn,
                           float *buffer,
                           uint16_t K,
                           bool transposed) {
  SdFile fi;
  fi.open(fn, O_READ);
  for (uint16_t i = 0; i < K; i++) {
    for (uint16_t j = 0; j < K; j++) {
      buffer[transposed ? j * K + i : i * K + j] = noodle_read_float(fi);
    }
  }
  fi.close();
}

void noodle_reset_buffer(float *buffer, uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    buffer[i] = 0.0;
  }
}

uint16_t noodle_conv(byte *grid,
                     float *output_buffer,
                     uint16_t n_inputs,
                     uint16_t n_outputs,
                     const char *in_fn,
                     const char *out_fn,
                     const char *weight_fn,
                     const char *bias_fn,
                     uint16_t W,
                     uint16_t P,
                     uint16_t K,
                     uint16_t S,
                     uint16_t M,
                     uint16_t T,
                     CBFPtr progress_cb) {
  float progress = 0;
  char i_fn[12], o_fn[12], w_fn[12];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  strcpy(w_fn, weight_fn);

  float kernel[K][K];
  SdFile fb;
  fb.open(bias_fn, O_READ);
  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(output_buffer, W * W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_n2ll(I, &i_fn[3]);
      noodle_n2ll(I, &w_fn[3]);
      noodle_n2ll(O, &w_fn[5]);
      noodle_grid_from_file(i_fn, grid, W, false);
      noodle_grid_from_file(w_fn, (float *)kernel, K, false);
      V = noodle_do_conv(grid, (float *)kernel, K, W, output_buffer, P, S);
      if (progress_cb) progress_cb(progress);
      progress += 1.0 / (float)(n_inputs * n_outputs - 1);
    }
    V = noodle_do_bias(output_buffer, bias, V);
    noodle_n2ll(O, &o_fn[3]);
    V = noodle_do_pooling(output_buffer, V, M, T, o_fn);
  }
  fb.close();
  return V;
}

uint16_t noodle_flat(const char *in_fn,
                     float *output_buffer,
                     uint16_t V,
                     uint16_t n_filters) {
  char i_fn[12];
  strcpy(i_fn, in_fn);
  SdFile fi;
  noodle_reset_buffer(output_buffer, V * V * n_filters);
  for (uint16_t k = 0; k < n_filters; k++) {
    noodle_n2ll(k, &i_fn[3]);

    fi.open(i_fn, O_READ);
    for (uint16_t i = 0; i < (V * V); i++) {
      output_buffer[i * n_filters + k] = noodle_read_float(fi);
    }
    fi.close();
  }
  return V * V * n_filters;
}

uint16_t noodle_fcn(int8_t *input_buffer,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu,
                    CBFPtr progress_cb) {
  float progress = 0.0;
  SdFile fw, fb, fo;
  fw.open(weight_fn, O_READ);
  fb.open(bias_fn, O_READ);
  fo.open(out_fn, O_WRITE | O_CREAT | O_TRUNC);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    Serial.println(h);
    for (uint16_t j = 0; j < n_inputs; j++)
      h += (float)input_buffer[j] * noodle_read_float(fw);
    if ((h < 0.0) && with_relu) h = 0.0;
    noodle_write_float(fo, h);
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / ((float)(n_outputs - 1));
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

uint16_t noodle_fcn(float *input_buffer,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu,
                    CBFPtr progress_cb) {
  float progress = 0.0;
  SdFile fw, fb, fo;
  fw.open(weight_fn, O_READ);
  fb.open(bias_fn, O_READ);
  fo.open(out_fn, O_WRITE | O_CREAT | O_TRUNC);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      h += input_buffer[j] * noodle_read_float(fw);
    if ((h < 0.0) && with_relu) h = 0.0;
    noodle_write_float(fo, h);
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / ((float)(n_outputs - 1));
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

uint16_t noodle_fcn(byte *input_buffer,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu,
                    CBFPtr progress_cb) {
  float progress = 0;
  SdFile fw, fb, fo;
  fw.open(weight_fn, O_READ);
  fb.open(bias_fn, O_READ);
  fo.open(out_fn, O_WRITE | O_CREAT | O_TRUNC);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      h += (float)input_buffer[j] * noodle_read_float(fw);
    if ((h < 0.0) && with_relu) h = 0.0;
    noodle_write_float(fo, h);
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / ((float)(n_outputs - 1));
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}


uint16_t noodle_fcn(byte *input_buffer,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output_buffer,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu,
                    CBFPtr progress_cb) {
  float progress = 0;
  SdFile fw, fb;
  fw.open(weight_fn, O_READ);
  fb.open(bias_fn, O_READ);

  for (uint16_t k = 0; k < n_outputs; k++) {
    output_buffer[k] = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      output_buffer[k] += (float)input_buffer[j] * noodle_read_float(fw);
    if ((output_buffer[k] < 0.0) && with_relu) output_buffer[k] = 0.0;
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / ((float)(n_outputs - 1));
  }

  fw.close();
  fb.close();
  return n_outputs;
}

uint16_t noodle_fcn(float *input_buffer,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output_buffer,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu,
                    CBFPtr progress_cb) {
  float progress = 0;
  SdFile fw, fb;
  fw.open(weight_fn, O_READ);
  fb.open(bias_fn, O_READ);

  for (uint16_t k = 0; k < n_outputs; k++) {
    output_buffer[k] = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      output_buffer[k] += input_buffer[j] * noodle_read_float(fw);
    if ((output_buffer[k] < 0.0) && with_relu) output_buffer[k] = 0.0;
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / ((float)(n_outputs - 1));
  }

  fw.close();
  fb.close();
  return n_outputs;
}

uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output_buffer,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu,
                    CBFPtr progress_cb) {
  float progress = 0;
  SdFile fi, fw, fb;
  fw.open(weight_fn, O_READ);
  fb.open(bias_fn, O_READ);

  for (uint16_t j = 0; j < n_outputs; j++) {
    output_buffer[j] = noodle_read_float(fb);
    fi.open(in_fn, O_READ);
    for (uint16_t k = 0; k < n_inputs; k++)
      output_buffer[j] += noodle_read_float(fi) * noodle_read_float(fw);
    if ((output_buffer[j] < 0.0) && with_relu) output_buffer[j] = 0.0;
    fi.close();
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / ((float)(n_outputs - 1));
  }

  fw.close();
  fb.close();
  return n_outputs;
}


uint16_t noodle_fcn(int8_t *input_buffer,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output_buffer,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu,
                    CBFPtr progress_cb) {
  float progress = 0;
  SdFile fw, fb;
  fw.open(weight_fn, O_READ);
  fb.open(bias_fn, O_READ);

  for (uint16_t j = 0; j < n_outputs; j++) {
    output_buffer[j] = noodle_read_float(fb);
    for (uint16_t k = 0; k < n_inputs; k++)
      output_buffer[j] += (float)input_buffer[k] * noodle_read_float(fw);
    if ((output_buffer[j] < 0.0) && with_relu) output_buffer[j] = 0.0;
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / ((float)(n_outputs - 1));
  }

  fw.close();
  fb.close();
  return n_outputs;
}

uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const char *weight_fn,
                    const char *bias_fn,
                    bool with_relu,
                    CBFPtr progress_cb) {
  float progress = 0;

  SdFile fw, fb, fo, fi;
  fw.open(weight_fn, O_READ);
  fb.open(bias_fn, O_READ);
  fo.open(out_fn, O_WRITE | O_CREAT | O_TRUNC);

  for (uint16_t j = 0; j < n_outputs; j++) {
    float h = noodle_read_float(fb);
    fi.open(in_fn, O_READ);
    for (uint16_t k = 0; k < n_inputs; k++)
      h += noodle_read_float(fi) * noodle_read_float(fw);
    if ((h < 0.0) && with_relu) h = 0.0;
    noodle_write_float(fo, h);
    fi.close();
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / ((float)(n_outputs - 1));
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

uint16_t noodle_soft_max(float *input_output, uint16_t n) {
  float max_val = input_output[0];
  for (int i = 1; i < n; i++) {
    if (input_output[i] > max_val)
      max_val = input_output[i];
  }

  float sum = 0.0;
  for (int i = 0; i < n; i++) {
    input_output[i] = expf(input_output[i] - max_val);
    sum += input_output[i];
  }

  for (int i = 0; i < n; i++) {
    input_output[i] /= sum;
  }
  return n;
}

uint16_t noodle_sigmoid(float *input_output, uint16_t n) {
  for (int i = 0; i < n; i++) {
    input_output[i] = 1.0f / (1.0f + expf(-input_output[i]));
  }
  return n;
}

uint16_t noodle_do_conv1d(float *input,
                          float *kernel,
                          uint16_t W,
                          uint16_t K,
                          float *output_buffer,
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
    output_buffer[i] += acc;
  }

  return V;
}

uint16_t noodle_do_pooling1d(float *buffer,
                             uint16_t W,
                             uint16_t K,
                             uint16_t S,
                             const char *fn) {
  SdFile fo;
  fo.open(fn, O_WRITE | O_CREAT | O_TRUNC);

  uint16_t Wo = (W - K) / S + 1;
  for (uint16_t i = 0; i < Wo; i++) {
    float v = 0.0;
    for (uint16_t k = 1; k < K; k++) {
      if (v < buffer[i * S + k]) v = buffer[i * S + k];
    }
    noodle_write_float(fo, v);
  }

  fo.close();
  return Wo;
}

uint16_t noodle_conv1d(float *input,
                       float *output_buffer,
                       uint16_t n_inputs,
                       uint16_t n_outputs,
                       const char *in_fn,
                       const char *out_fn,
                       const char *weight_fn,
                       const char *bias_fn,
                       uint16_t W,
                       uint16_t P,
                       uint16_t K,
                       uint16_t S,
                       uint16_t M,
                       uint16_t T,
                       bool with_relu,
                       CBFPtr progress_cb) {
  float progress = 0;
  char i_fn[12], o_fn[12], w_fn[12];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  strcpy(w_fn, weight_fn);

  float kernel[K];
  SdFile fi, fb, fw;

  uint16_t V = 0;
  fb.open(bias_fn, O_READ);
  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(output_buffer, W);
    float bias = noodle_read_float(fb);

    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_n2ll(I, &i_fn[3]);
      noodle_n2ll(I, &w_fn[3]);
      noodle_n2ll(O, &w_fn[5]);

      fi.open(i_fn, O_READ);
      for (uint16_t i = 0; i < W; i++) {
        input[i] = noodle_read_float(fi);
      }
      fi.close();

      fw.open(w_fn, O_READ);
      for (uint16_t i = 0; i < K; i++) {
        kernel[i] = noodle_read_float(fw);
      }
      fw.close();

      V = noodle_do_conv1d(input, kernel, W, K, output_buffer, P, S);

      if (progress_cb) progress_cb(progress);
      progress += 1.0 / (float)(n_inputs * n_outputs - 1);
    }

    for (uint16_t i = 0; i < V; i++) {
      output_buffer[i] += bias;
      if ((output_buffer[i]) < 0.0 && with_relu) output_buffer[i] = 0.0;
    }

    noodle_n2ll(O, &o_fn[3]);
    V = noodle_do_pooling1d(output_buffer, V, M, T, o_fn);
  }

  fb.close();
  return V;
}

uint16_t noodle_conv1d(float *input,
                       float *output_buffer,
                       uint16_t n_inputs,
                       uint16_t n_outputs,
                       const char *in_fn,
                       const char *out_fn,
                       const char *weight_fn,
                       const char *bias_fn,
                       uint16_t W,
                       uint16_t P,
                       uint16_t K,
                       uint16_t S,
                       bool with_relu,
                       CBFPtr progress_cb) {

  float progress = 0;
  char i_fn[12], o_fn[12], w_fn[12];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  strcpy(w_fn, weight_fn);

  float kernel[K];
  SdFile fb;
  fb.open(bias_fn, O_READ);
  uint16_t V = 0;

  SdFile fi, fw;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(output_buffer, W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_n2ll(I, &i_fn[3]);
      noodle_n2ll(I, &w_fn[3]);
      noodle_n2ll(O, &w_fn[5]);

      fi.open(i_fn, O_READ);
      for (uint16_t i = 0; i < W; i++) {
        input[i] = noodle_read_float(fi);
      }
      fi.close();

      fw.open(w_fn, O_READ);
      for (uint16_t i = 0; i < K; i++) {
        kernel[i] = noodle_read_float(fw);
      }
      fw.close();

      V = noodle_do_conv1d(input, kernel, W, K, output_buffer, P, S);

      if (progress_cb) progress_cb(progress);
      progress += 1.0 / (float)(n_inputs * n_outputs - 1);
    }

    for (uint16_t i = 0; i < V; i++) {
      output_buffer[i] += bias;
      if ((output_buffer[i]) < 0.0 && with_relu) output_buffer[i] = 0.0;
    }

    noodle_n2ll(O, &o_fn[3]);
    noodle_array_to_file(output_buffer, o_fn, V);
  }

  fb.close();
  return V;
}