/**
 * @file noodle_internal.cpp
 * @brief Private shared globals and helpers for Noodle implementation files.
 */
#include "noodle_internal.h"


#if defined(NOODLE_USE_SDFAT)
SdFat NOODLE_FS;  // define the SdFat object declared in noodle_fs.h
#endif

// File handles and temp buffers (backend-agnostic)
NDL_File fw, fb, fo, fi;
void *temp_buff1 = NULL;
void *temp_buff2 = NULL;

// ===== Convolution private helpers moved from noodle_conv.cpp =====

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
  fo = noodle_fs_open_write(fn);

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

uint16_t noodle_do_pooling1d(const float *input,
                             uint16_t W,
                             uint16_t K,
                             uint16_t S,
                             float *output) {
  if (K <= 1) {
    for (uint16_t i = 0; i < W; i++) {
      output[i] = input[i];
    }
    return W;
  }

  if (S == 0) S = K;

  const uint16_t Wo = (uint16_t)((W - K) / S + 1);

  for (uint16_t i = 0; i < Wo; i++) {
    const uint16_t base = (uint16_t)(i * S);

#if NOODLE_POOL_MODE == NOODLE_POOL_MEAN
    float acc = 0.0f;
    for (uint16_t k = 0; k < K; k++) {
      acc += input[base + k];
    }
    output[i] = acc / (float)K;
#else
    float v = input[base];

    for (uint16_t k = 1; k < K; k++) {
      const float u = input[base + k];
      if (u > v) v = u;
    }

    output[i] = v;
#endif
  }

  return Wo;
}

float noodle_get_padded_x(byte *grid,
                          int16_t i, int16_t j,
                          int16_t W,
                          int16_t P0, int16_t P1) {
  const int16_t Wpad = W + P0 + P1;

  // outside padded tensor, fast reject using padded bounds
  if ((uint16_t)i >= (uint16_t)Wpad || (uint16_t)j >= (uint16_t)Wpad)
    return 0.0f;       

  // padded -> input
  const int16_t ii = i - P0;
  const int16_t jj = j - P0;

  if (ii < 0 || jj < 0 || ii >= W || jj >= W) return 0.0f;

  return (float)grid[ii * W + jj];
}

float noodle_get_padded_x(float *grid,
                          int16_t i, int16_t j,
                          int16_t W,
                          int16_t P0, int16_t P1) {
  const int16_t Wpad = W + P0 + P1;

  // outside padded tensor, fast reject using padded bounds
  if ((uint16_t)i >= (uint16_t)Wpad || (uint16_t)j >= (uint16_t)Wpad)
    return 0.0f;                          

  // padded -> input
  const int16_t ii = i - P0;
  const int16_t jj = j - P0;

  if (ii < 0 || jj < 0 || ii >= W || jj >= W) return 0.0f;

  return grid[ii * W + jj];
}

uint16_t noodle_do_bias(float *output,
                        float bias,
                        uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    const uint16_t row = i * n;
    for (uint16_t j = 0; j < n; j++) {
      const uint16_t idx = row + j;
      output[idx] += bias;
      if (output[idx] < 0.0f) {
        output[idx] = 0.0f;
      }
    }
  }
  return n;
}

uint16_t noodle_do_pooling(const float *input,
                           uint16_t W,
                           uint16_t K,
                           uint16_t S,
                           const char *fn) {

#if NOODLE_POOL_MODE == NOODLE_POOL_NONE
  fo = noodle_fs_open_write(fn);
  const uint32_t n = (uint32_t)W * (uint32_t)W;
  for (uint32_t i = 0; i < n; i++)
    noodle_write_float(fo, input[i]);
  fo.close();
  return W;

#else
  if (S == 0) return 0;
  if (W < K) return 0;

  const uint16_t Wo = (uint16_t)((W - K) / S + 1);
  fo = noodle_fs_open_write(fn);

  #if NOODLE_POOL_MODE == NOODLE_POOL_MEAN
    const float inv_KK = 1.0f / (float)(K * K);
  #endif

  for (uint16_t out_y = 0; out_y < Wo; out_y++) {
    const uint16_t base_y = (uint16_t)(out_y * S);
    for (uint16_t out_x = 0; out_x < Wo; out_x++) {
      const uint16_t base_x = (uint16_t)(out_x * S);

    #if NOODLE_POOL_MODE == NOODLE_POOL_MAX
      float vmax = -FLT_MAX;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (uint16_t)((base_y + win_y) * W);
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          const float v = input[row + base_x + win_x];
          if (v > vmax) vmax = v;
        }
      }
      noodle_write_float(fo, vmax);

    #elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
      float acc = 0.0f;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (uint16_t)((base_y + win_y) * W);
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
#endif
}

uint16_t noodle_do_pooling(const float *input,
                           uint16_t W,
                           uint16_t K,
                           uint16_t S,
                           NDL_File &fo) {

#if NOODLE_POOL_MODE == NOODLE_POOL_NONE
  // Identity pooling: write W*W values unchanged.
  const uint32_t n = (uint32_t)W * (uint32_t)W;
  for (uint32_t i = 0; i < n; i++) {
    noodle_write_float(fo, input[i]);
  }
  return W;

#else
  if (S == 0) return 0;
  if (W < K)  return 0;

  const uint16_t Wo = (uint16_t)((W - K) / S + 1);

  #if NOODLE_POOL_MODE == NOODLE_POOL_MEAN
    const float inv_KK = 1.0f / (float)(K * K);
  #endif

  for (uint16_t out_y = 0; out_y < Wo; out_y++) {
    const uint16_t base_y = (uint16_t)(out_y * S);
    for (uint16_t out_x = 0; out_x < Wo; out_x++) {
      const uint16_t base_x = (uint16_t)(out_x * S);

    #if NOODLE_POOL_MODE == NOODLE_POOL_MAX
      float vmax = -FLT_MAX;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (uint16_t)((base_y + win_y) * W);
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          const float v = input[row + base_x + win_x];
          if (v > vmax) vmax = v;
        }
      }
      noodle_write_float(fo, vmax);

    #elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
      float acc = 0.0f;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (uint16_t)((base_y + win_y) * W);
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          acc += input[row + base_x + win_x];
        }
      }
      noodle_write_float(fo, acc * inv_KK);
    #endif
    }
  }

  return Wo;
#endif
}

uint16_t noodle_do_pooling(const float *input,
                           uint16_t W,
                           uint16_t K,
                           uint16_t S,
                           float *output) {

#if NOODLE_POOL_MODE == NOODLE_POOL_NONE
  // Identity pooling: copy W*W values unchanged.
  const uint32_t n = (uint32_t)W * (uint32_t)W;
  for (uint32_t i = 0; i < n; i++) {
    output[i] = input[i];
  }
  return W;

#else
  if (K == 1 && S == 1) { // identity, skip pooling!
    const uint32_t n = (uint32_t)W * (uint32_t)W;
    for (uint32_t i = 0; i < n; i++) output[i] = input[i];
    return W;
  }

  const uint16_t Wo = (uint16_t)((W - K) / S + 1);

  #if NOODLE_POOL_MODE == NOODLE_POOL_MEAN
    const float inv_KK = 1.0f / (float)(K * K);
  #endif

  for (uint16_t out_y = 0; out_y < Wo; out_y++) {
    const uint16_t base_y = (uint16_t)(out_y * S);
    const uint16_t out_row = (uint16_t)(out_y * Wo);

    for (uint16_t out_x = 0; out_x < Wo; out_x++) {
      const uint16_t base_x = (uint16_t)(out_x * S);
      const uint16_t out_idx = (uint16_t)(out_row + out_x);

    #if NOODLE_POOL_MODE == NOODLE_POOL_MAX
      float vmax = -FLT_MAX;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (uint16_t)((base_y + win_y) * W);
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          const float v = input[row + base_x + win_x];
          if (v > vmax) vmax = v;
        }
      }
      output[out_idx] = vmax;

    #elif NOODLE_POOL_MODE == NOODLE_POOL_MEAN
      float acc = 0.0f;
      for (uint16_t win_y = 0; win_y < K; win_y++) {
        const uint16_t row = (uint16_t)((base_y + win_y) * W);
        for (uint16_t win_x = 0; win_x < K; win_x++) {
          acc += input[row + base_x + win_x];
        }
      }
      output[out_idx] = acc * inv_KK;
    #endif
    }
  }

  return Wo;
#endif
}

uint16_t noodle_do_conv(byte *grid,
                        const float *kernel,
                        uint16_t K,
                        uint16_t W,
                        float *output,
                        uint16_t P,
                        uint16_t S) {
  uint16_t P0, P1;
  uint16_t V = noodle_compute_V_and_P(K, W, P, S, P0, P1);

  for (uint16_t i = 0; i < V; i++) {
    for (uint16_t j = 0; j < V; j++) {
      float v = 0.0f;
      for (uint16_t k = 0; k < K; k++) {
        for (uint16_t l = 0; l < K; l++) {
          v += kernel[k * K + l] *
               noodle_get_padded_x(grid, i * S + k, j * S + l, W, P0, P1);
        }
      }
      output[i * V + j] += v;
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
  uint16_t P0, P1;
  uint16_t V = noodle_compute_V_and_P(K, W, P, S, P0, P1);

  for (uint16_t i = 0; i < V; i++) {
    for (uint16_t j = 0; j < V; j++) {
      float v = 0.0f;
      for (uint16_t k = 0; k < K; k++) {
        for (uint16_t l = 0; l < K; l++) {
          v += kernel[k * K + l] *
               noodle_get_padded_x(grid, i * S + k, j * S + l, W, P0, P1);
        }
      }
      output[i * V + j] += v;
    }
  }

  return V;
}

void noodle_reset_buffer(float *buffer,
                         uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    buffer[i] = 0.0;
  }
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

uint16_t noodle_do_conv_transpose(float *input,
                                  const float *kernel,
                                  uint16_t K,
                                  uint16_t W,
                                  float *output,
                                  uint16_t P,
                                  uint16_t S,
                                  uint16_t OP) {
  uint16_t P0, P1;
  const uint16_t Vt = noodle_compute_Vt_and_P(K, W, P, S, OP, P0, P1);

  if (Vt == 0) return 0;

  // Scatter each input pixel through the KxK kernel into the expanded output.
  for (uint16_t iy = 0; iy < W; iy++) {
    for (uint16_t ix = 0; ix < W; ix++) {
      const float x = input[(uint32_t)iy * W + ix];
      if (x == 0.0f) continue;

      const int32_t base_y = (int32_t)iy * (int32_t)S - (int32_t)P0;
      const int32_t base_x = (int32_t)ix * (int32_t)S - (int32_t)P0;

      for (uint16_t ky = 0; ky < K; ky++) {
        const int32_t oy = base_y + ky;
        if ((uint32_t)oy >= (uint32_t)Vt) continue;

        for (uint16_t kx = 0; kx < K; kx++) {
          const int32_t ox = base_x + kx;
          if ((uint32_t)ox >= (uint32_t)Vt) continue;

          output[(uint32_t)oy * Vt + ox] +=
              x * kernel[(uint32_t)ky * K + kx];
        }
      }
    }
  }

  return Vt;
}

uint16_t noodle_compute_V(uint16_t K,
                          uint16_t W, 
                          uint16_t P, 
                          uint16_t S){
  uint16_t P0, P1;
  return noodle_compute_V_and_P(K, W, P, S, P0, P1);
}

uint16_t noodle_compute_V_and_P(uint16_t K,
                                uint16_t W,
                                uint16_t P,
                                uint16_t S,
                                uint16_t &P0,
                                uint16_t &P1) {
  uint16_t V;

  if (P == 65535) {
    // TF/Keras SAME
    V = (W + S - 1) / S;   // ceil(W / S)

    int32_t Ptot = (int32_t)(V - 1) * (int32_t)S + (int32_t)K - (int32_t)W;
    if (Ptot < 0) Ptot = 0;

    P0 = (uint16_t)(Ptot / 2);
    P1 = (uint16_t)(Ptot - P0);
  } else {
    // explicit symmetric padding
    P0 = P; // top/left
    P1 = P; // bottom/right
    V = (W - K + 2 * P) / S + 1;
  }

  return V;
}

uint16_t noodle_valid_max_pool(float *inplace,
                               uint16_t W,
                               uint16_t C,
                               const Pool &pool)
{
  const uint16_t M = pool.M;  // Kernel
  const uint16_t T = pool.T;  // Stride

  if (!inplace) return 0;
  if (W == 0 || C == 0 || M == 0 || T == 0) return 0;
  if (M > W) return 0;

  const uint16_t Wo = (W - M) / T + 1;

  // Identity: M=1, T=1 => nothing changes; still stacked as W*W per channel
  if (M == 1 && T == 1) return W;

  const uint16_t in_ch_stride  = W  * W;   // floats per input channel
  const uint16_t out_ch_stride = Wo * Wo;  // floats per output channel

  for (uint16_t c = 0; c < C; c++) {
    float *in_ch  = inplace + c * in_ch_stride;
    float *out_ch = inplace + c * out_ch_stride;

    for (uint16_t oy = 0; oy < Wo; oy++) {
      const uint16_t base_y = oy * T;
      const uint16_t out_row = oy * Wo;

      for (uint16_t ox = 0; ox < Wo; ox++) {
        const uint16_t base_x = ox * T;

        float vmax = -FLT_MAX;

        for (uint16_t ky = 0; ky < M; ky++) {
          const uint16_t row = (base_y + ky) * W;
          const uint16_t idx0 = row + base_x;

          for (uint16_t kx = 0; kx < M; kx++) {
            const float v = in_ch[idx0 + kx];
            if (v > vmax) vmax = v;
          }
        }

        out_ch[out_row + ox] = vmax;
      }
    }
  }

  return Wo;
}

uint16_t noodle_compute_Vt(uint16_t K,
                           uint16_t W,
                           uint16_t P,
                           uint16_t S,
                           uint16_t OP) {
  uint16_t P0, P1;
  return noodle_compute_Vt_and_P(K, W, P, S, OP, P0, P1);
}

uint16_t noodle_compute_Vt_and_P(uint16_t K,
                                 uint16_t W,
                                 uint16_t P,
                                 uint16_t S,
                                 uint16_t OP,
                                 uint16_t &P0,
                                 uint16_t &P1) {
  if (W == 0 || K == 0 || S == 0) return 0;

  if (P == 65535) {
    // Keras/TF Conv2DTranspose padding="same"
    //
    // Output size:
    //   Vt = W * S
    //
    // Full scatter size:
    //   Vfull = (W - 1)*S + K
    //
    // Crop:
    //   crop_total = Vfull - Vt = K - S
    //
    // For K=3, S=2:
    //   crop_total = 1
    //   P0 = 0
    //   P1 = 1
    //
    // This means crop only bottom/right, matching Keras-style SAME.
    const int32_t Vt = (int32_t)W * (int32_t)S;

    int32_t crop_total = ((int32_t)W - 1) * (int32_t)S
                       + (int32_t)K
                       - Vt;

    if (crop_total < 0) crop_total = 0;

    P0 = (uint16_t)(crop_total / 2);
    P1 = (uint16_t)(crop_total - P0);

    if (Vt <= 0 || Vt > 65535) return 0;
    return (uint16_t)Vt;
  }

  // Explicit symmetric transpose padding, old Noodle behavior.
  P0 = P;
  P1 = P;

  const int32_t vt = ((int32_t)W - 1) * (int32_t)S
                   + (int32_t)K
                   - (int32_t)P0
                   - (int32_t)P1
                   + (int32_t)OP;

  if (vt <= 0) return 0;
  if (vt > 65535) return 0;

  return (uint16_t)vt;
}

void noodle_copy_kernel_progmem(const float *w,
                                uint32_t base,
                                uint16_t K,
                                float *kernel) {
  const uint16_t KK = (uint16_t)(K * K);
  for (uint16_t i = 0; i < KK; i++) {
    kernel[i] = noodle_pgm_float(w, base + i);
  }
}
