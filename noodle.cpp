#include "noodle.h"

SdFat SD;

float noodle_read_float(File &f) {
  char s[20];
  f.readBytesUntil('\n', (char *)s, sizeof(s));
  return atof(s);
}

void noodle_delete_file(char *fn) {
  SD.remove(fn);
}

float *noodle_create_buffer(uint16_t size) {
  return (float *)malloc(size);
}

void noodle_delete_buffer(float *buffer) {
  free(buffer);
}

void noodle_grid_to_file(byte *grid, char *fn, uint16_t n) {
  noodle_delete_file(fn);
  File fo = SD.open(fn, FILE_WRITE);
  for (int16_t i = 0; i < n; i++)
    for (int16_t j = 0; j < n; j++) {
      fo.print(grid[i * n + j]);
      fo.println('\0');
    }
  fo.close();
}

float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P) {
  if ((i < P) || (j < P) || (i > (W - 1 + P)) || (j > (W - 1 + P)))
    return 0.0;
  return (float)grid[(i - P) * W + (j - P)];
}

uint16_t noodle_do_bias(float *output, float bias, uint16_t n) {
  for (int16_t i = 0; i < n; i++) {
    for (int16_t j = 0; j < n; j++) {
      output[i * n + j] += bias;
      if (output[i * n + j] < 0.0)
        output[i * n + j] = 0.0;
    }
  }
  return n;
}

uint16_t noodle_do_pooling(float *buffer, uint16_t W, uint16_t K, uint16_t S, char *fn) {
  uint16_t Wo = (W - K) / S + 1;
  noodle_delete_file(fn);
  File fo = SD.open(fn, FILE_WRITE);

  for (int16_t i = 0; i < Wo; i++) {
    for (int16_t j = 0; j < Wo; j++) {
      float v = 0.0;
      for (int16_t k = 0; k < K; k++)
        for (int16_t l = 0; l < K; l++)
          if (v < buffer[(i * S + k) * W + (j * S + l)])
            v = buffer[(i * S + k) * W + (j * S + l)];

      buffer[i * Wo + j] = v;
      fo.print(buffer[i * Wo + j], 8);
      fo.println('\0');
    }
  }
  fo.close();
  return Wo;
}

uint16_t noodle_do_convolution(byte *grid, float *kernel, uint16_t K, uint16_t W, float *output_buffer, uint16_t P, uint16_t S) {
  uint16_t V = (W - K + 2 * P) / S + 1;
  for (int16_t i = 0; i < V; i++) {
    for (int16_t j = 0; j < V; j++) {
      float v = 0;
      for (int16_t k = 0; k < K; k++)
        for (int16_t l = 0; l < K; l++)
          v += kernel[k * K + l] * noodle_get_padded_x(grid, i * S + k, j * S + l, W, P);
      output_buffer[i * V + j] += v;
    }
  }
  return V;
}

void noodle_read_from_file(char *fn, byte *buffer, uint16_t K, bool transposed) {
  File fi = SD.open(fn, FILE_READ);
  for (uint16_t i = 0; i < K; i++)
    for (uint16_t j = 0; j < K; j++)
      buffer[transposed ? j * K + i : i * K + j] = noodle_read_float(fi);
  fi.close();
}

void noodle_read_from_file(char *fn, float *buffer, uint16_t K, bool transposed) {
  File fi = SD.open(fn, FILE_READ);
  for (uint16_t i = 0; i < K; i++)
    for (uint16_t j = 0; j < K; j++)
      buffer[transposed ? j * K + i : i * K + j] = noodle_read_float(fi);
  fi.close();
}

void noodle_reset_buffer(float *buffer, uint16_t n) {
  for (uint16_t i = 0; i < n; i++)
    buffer[i] = 0.0;
}

uint16_t noodle_conv(byte *grid, float *output_buffer, uint16_t n_inputs, uint16_t n_outputs, char *in_fn, char *out_fn, char *weight_fn, char *bias_fn, uint16_t W, uint16_t P, uint16_t K, uint16_t S, uint16_t M, uint16_t T, CBFPtr progress_cb) {
  float progress = 0;
  char i_fn[12], o_fn[12], w_fn[12];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  strcpy(w_fn, weight_fn);

  float kernel[K][K];
  File fb = SD.open(bias_fn, FILE_READ);
  uint16_t V;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(output_buffer, W * W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {
      i_fn[3] = I + 'a';
      w_fn[3] = I + 'a';
      w_fn[5] = O + 'a';
      noodle_read_from_file(i_fn, grid, W, false);
      noodle_read_from_file(w_fn, (float *)kernel, K, false);
      V = noodle_do_convolution(grid, (float *)kernel, K, W, output_buffer, P, S);
      if (progress_cb) progress_cb(progress);
      progress += 1.0 / (float)(n_inputs * n_outputs) + 1.0;
    }
    V = noodle_do_bias(output_buffer, bias, V);
    o_fn[3] = O + 'a';
    V = noodle_do_pooling(output_buffer, V, M, T, o_fn);
  }
  fb.close();
  return V;
}

uint16_t noodle_flat(char *in_fn, float *output_buffer, uint16_t V, uint16_t n_filters) {
  char i_fn[12];
  strcpy(i_fn, in_fn);
  File fi;
  noodle_reset_buffer(output_buffer, V * V * n_filters);
  for (uint16_t k = 0; k < n_filters; k++) {
    i_fn[3] = k + 'a';
    fi = SD.open(i_fn, FILE_READ);
    for (uint16_t i = 0; i < (V * V); i++)
      output_buffer[i * n_filters + k] = noodle_read_float(fi);
    fi.close();
  }
  return V * V * n_filters;
}

uint16_t noodle_fcn(float *input_buffer, uint16_t n_inputs, uint16_t n_outputs, char *out_fn, char *weight_fn, char *bias_fn, CBFPtr progress_cb) {
  float progress = 0;
  File fw = SD.open(weight_fn, FILE_READ);
  File fb = SD.open(bias_fn, FILE_READ);
  noodle_delete_file(out_fn);
  File fo = SD.open(out_fn, FILE_WRITE);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      h += input_buffer[j] * noodle_read_float(fw);
    if (h < 0.0) h = 0.0;
    fo.print(h, 8);
    fo.println('\0');
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / (float)n_outputs;
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

uint16_t noodle_fcn(byte *input_buffer, uint16_t n_inputs, uint16_t n_outputs, char *out_fn, char *weight_fn, char *bias_fn, CBFPtr progress_cb) {
  float progress = 0;
  File fw = SD.open(weight_fn, FILE_READ);
  File fb = SD.open(bias_fn, FILE_READ);
  noodle_delete_file(out_fn);
  File fo = SD.open(out_fn, FILE_WRITE);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      h += (float)input_buffer[j] * noodle_read_float(fw);
    if (h < 0.0) h = 0.0;
    fo.print(h, 8);
    fo.println('\0');
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / (float)n_outputs;
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}


uint16_t noodle_fcn(byte *input_buffer, uint16_t n_inputs, uint16_t n_outputs, float *output_buffer, char *weight_fn, char *bias_fn, CBFPtr progress_cb) {
  float progress = 0;
  File fw = SD.open(weight_fn, FILE_READ);
  File fb = SD.open(bias_fn, FILE_READ);
 
  for (uint16_t k = 0; k < n_outputs; k++) {
    output_buffer[k] = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      output_buffer[k] += (float)input_buffer[j] * noodle_read_float(fw);
    if (output_buffer[k] < 0.0) output_buffer[k] = 0.0;
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / (float)n_outputs;
  }

  fw.close();
  fb.close();
  return n_outputs;
}

uint16_t noodle_fcn(float *input_buffer, uint16_t n_inputs, uint16_t n_outputs, float *output_buffer, char *weight_fn, char *bias_fn, CBFPtr progress_cb) {
  float progress = 0;
  File fw = SD.open(weight_fn, FILE_READ);
  File fb = SD.open(bias_fn, FILE_READ);
 
  for (uint16_t k = 0; k < n_outputs; k++) {
    output_buffer[k] = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      output_buffer[k] += input_buffer[j] * noodle_read_float(fw);
    if (output_buffer[k] < 0.0) output_buffer[k] = 0.0;
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / (float)n_outputs;
  }

  fw.close();
  fb.close();
  return n_outputs;
}

uint16_t noodle_fcn(char *in_fn, uint16_t n_inputs, uint16_t n_outputs, float *output_buffer, char *weight_fn, char *bias_fn, CBFPtr progress_cb) {
  float progress = 0;
  File fi, fw, fb;
  fw = SD.open(weight_fn, FILE_READ);
  fb = SD.open(bias_fn, FILE_READ);

  for (uint16_t j = 0; j < n_outputs; j++) {
    output_buffer[j] = noodle_read_float(fb);
    fi = SD.open(in_fn, FILE_READ);
    for (uint16_t k = 0; k < n_inputs; k++)
      output_buffer[j] += noodle_read_float(fi) * noodle_read_float(fw);
    fi.close();
    if (progress_cb) progress_cb(progress);
    progress += 1.0 / (float)n_outputs;
  }

  fw.close();
  fb.close();
  return n_outputs;
}
