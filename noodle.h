/**
 * Auralius Manurung -- Universitas Telkom, Bandung
 *   auralius.manurung@ieee.org
 * Lisa Kristiana -- ITENAS, Bandung
 *   lisa@itenas.ac.id
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include <SdFat.h>
SdFat SD;

typedef void (*CBFPtr)(float);

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
      fo.print(grid[i * n + j]);  //transposed
      fo.println('\0');
    }
  fo.close();
}


float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P) {
  if ((i < P) || (j < P) || (i > (W - 1 + P)) || (j > (W - 1 + P)))
    return 0.0;

  return (float)grid[(i - P) * W + (j - P)];
}


// Bias with ReLU
uint16_t noodle_do_bias(float *output, float bias, uint16_t n) {
  for (int16_t i = 0; i < n; i++) {
    for (int16_t j = 0; j < n; j++) {
      output[i * n + j] = output[i * n + j] + bias;
      if (output[i * n + j] < 0.0)
        output[i * n + j] = 0.0;
    }
  }
  return n;
}


// Do pooling and strore the result
uint16_t noodle_do_pooling(float *output, uint16_t W, uint16_t K, uint16_t S, char *fn) {
  uint16_t Wo = (W - K) / S + 1;
  noodle_delete_file(fn);
  File fo = SD.open(fn, FILE_WRITE);

  // Max Pooling
  for (int16_t i = 0; i < Wo; i++) {
    for (int16_t j = 0; j < Wo; j++) {
      float v = 0.0;
      for (int16_t k = 0; k < K; k++)
        for (int16_t l = 0; l < K; l++)
          if (v < output[(i * S + k) * W + (j * S + l)])
            v = output[(i * S + k) * W + (j * S + l)];

      output[i * Wo + j] = v;

      fo.print(output[i * Wo + j], 8);
      fo.println('\0');
    }
  }
  fo.close();

  return Wo;
}


// Input size is W x W.
// The kernel filter size is K x K.
// The padding is P (uniform and zero padding).
// The stride length is S
uint16_t noodle_do_convolution(byte *grid, float *kernel, uint16_t K, uint16_t W, float *output_buffer, uint16_t P, uint16_t S) {
  // Caclulate the output size
  uint16_t V = (W - K + 2 * P) / S + 1;

  for (int16_t i = 0; i < V; i++) {
    for (int16_t j = 0; j < V; j++) {
      float v = 0;
      for (int16_t k = 0; k < K; k++)
        for (int16_t l = 0; l < K; l++)
          v = v + kernel[k * K + l] * noodle_get_padded_x(grid, i * S + k, j * S + l, W, P);
      output_buffer[i * V + j] = output_buffer[i * V + j] + v;
    }
  }

  return V;
}


// Load a BYTE  square matrix from a file (K x K).
// The matrix was previously stored linearly
void noodle_read_from_file(char *fn, byte *buffer, uint16_t K, bool transposed = false) {
  File fi;
  fi = SD.open(fn, FILE_READ);

  for (uint16_t i = 0; i < K; i++)
    for (uint16_t j = 0; j < K; j++) {
      if (transposed)
        buffer[j * K + i] = noodle_read_float(fi);
      else
        buffer[i * K + j] = noodle_read_float(fi);
    }

  fi.close();
}


// Load a FLOAT square matrix from a file (K x K).
// The matrix was previously stored linearly
void noodle_read_from_file(char *fn,
                           float *buffer,
                           uint16_t K,
                           bool transposed = false) {
  File fi;
  fi = SD.open(fn, FILE_READ);

  for (uint16_t i = 0; i < K; i++)
    for (uint16_t j = 0; j < K; j++)
      if (!transposed)
        buffer[i * K + j] = noodle_read_float(fi);
      else
        buffer[j * K + i] = noodle_read_float(fi);
  fi.close();
}


void noodle_reset_buffer(float *buffer, uint16_t n) {
  for (uint16_t i = 0; i < n; i++)
    buffer[i] = 0.0;
}


uint16_t noodle_conv(byte *grid,
                     float *output_buffer,
                     uint16_t n_inputs,
                     uint16_t n_filters,
                     char *in_fn, char *out_fn,
                     char *weight_fn, char *bias_fn,
                     uint16_t W,  // Input size (W x W)
                     uint16_t P,  // Number of uniform-zero-padding layer for the input
                     uint16_t K,  // Convolution kernel size (K x K)
                     uint16_t S,  // Convolution stride length
                     uint16_t M,  // Max-pooling filter size
                     uint16_t T,  // Max-poooling stride length
                     CBFPtr progress_cb=NULL)  
{
  float progress = 0;

  char i_fn[12];
  char o_fn[12];
  char w_fn[12];
  strcpy(i_fn, in_fn);
  strcpy(o_fn, out_fn);
  strcpy(w_fn, weight_fn);

  File fb, fo;
  float kernel[5][5];
  uint16_t V;

  fb = SD.open(bias_fn, FILE_READ);
  for (uint16_t O = 0; O < n_filters; O++) {
    noodle_reset_buffer(output_buffer, W * W);
    float bias = noodle_read_float(fb);
    for (uint16_t I = 0; I < n_inputs; I++) {
      i_fn[3] = I + 'a';
      w_fn[3] = I + 'a';
      w_fn[5] = O + 'a';
      noodle_read_from_file(i_fn, grid, W);
      noodle_read_from_file(w_fn, (float *)kernel, K);
      V = noodle_do_convolution(grid, (float *)kernel, K, W, output_buffer, P, S);

      if (progress_cb)
        progress_cb(progress);
      progress = progress + 1.0 / (float)(n_inputs * n_filters);

    }

    V = noodle_do_bias(output_buffer, bias, V);

    o_fn[3] = O + 'a';
    V = noodle_do_pooling(output_buffer, V, M, T, o_fn);
  }

  fb.close();

  return V;
}

// Flattening, from a several input files to output_buffer
uint16_t noodle_flat(float *output_buffer, 
                     char *in_fn, 
                     uint16_t V, 
                     uint16_t n_filters) {
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


// From output_buffer to out_fn
uint16_t noodle_fcn(float *output_buffer,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    char *out_fn,
                    char *weight_fn,
                    char *bias_fn,
                    CBFPtr progress_cb=NULL) {
  float progress = 0;

  File fw, fo, fb;
  fw = SD.open(weight_fn, FILE_READ);
  fb = SD.open(bias_fn, FILE_READ);

  noodle_delete_file(out_fn);
  fo = SD.open(out_fn, FILE_WRITE);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      h = h + output_buffer[j] * noodle_read_float(fw);

    if (h < 0.0)
      h = 0.0;

    fo.print(h, 8);
    fo.println('\0');

    if (progress_cb)
      progress_cb(progress);
    progress = progress + 1.0 / (float)n_outputs;
  }

  fo.close();
  fw.close();
  fb.close();

  return n_outputs;
}

// From in_fn to output_buffer
uint16_t noodle_fcn(char *in_fn, 
                    uint16_t n_inputs, 
                    uint16_t n_outputs, 
                    float *output_buffer, 
                    char *weight_fn, 
                    char *bias_fn,
                    CBFPtr progress_cb=NULL) {
  float progress = 0;
  File fi, fw, fb;

  fw = SD.open(weight_fn, FILE_READ);
  fb = SD.open(bias_fn, FILE_READ);

  for (uint16_t j = 0; j < n_outputs; j++) {
    output_buffer[j] = noodle_read_float(fb);
    fi = SD.open(in_fn, FILE_READ);
    for (uint16_t k = 0; k < n_inputs; k++)
      output_buffer[j] = output_buffer[j] + noodle_read_float(fi) * noodle_read_float(fw);
    fi.close();

    if (progress_cb)
      progress_cb(progress);
    progress = progress + 1.0 / (float)n_outputs;
  }

  fw.close();
  fb.close();

  return n_outputs;
}
