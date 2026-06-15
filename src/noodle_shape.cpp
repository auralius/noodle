/**
 * @file noodle_shape.cpp
 * @brief Tensor-to-vector transforms: flatten, reshape, GAP, and GMP.
 * @ingroup noodle_api
 */
#include "noodle_internal.h"


uint16_t noodle_flat(const char *in_fn,
                     float *output,
                     uint16_t V,
                     uint16_t n_filters) {
  fi = noodle_fs_open_read(in_fn);

  const uint16_t plane = V * V;

  // Input file is CHW: [ch0 plane][ch1 plane]...[chK plane]
  // Output wants HWC-flatten: output[i*n_filters + k]
  for (uint16_t k = 0; k < n_filters; k++) {
    for (uint16_t i = 0; i < plane; i++) {
      float x = noodle_read_float(fi);
      output[i * n_filters + k] = x;
    }
  }

  fi.close();
  // NOTE: return type is uint16_t for compatibility; caller should ensure plane*n_filters <= 65535 if it matters.
  return (plane * n_filters);
}


// Memory CHW → Memory HWC-flatten
uint16_t noodle_flat(float *input,
                     float *output,
                     uint16_t V,
                     uint16_t n_filters) {
  const uint16_t plane = V * V;
  for (uint16_t k = 0; k < n_filters; k++) {
    float *sliced_input = noodle_slice(input, V, k);
    for (uint16_t i = 0; i < plane; i++) {
      output[i * n_filters + k] = sliced_input[i];
    }
  }
  return (plane * n_filters);
}

uint16_t noodle_reshape(const float *src_hwc, 
                        float *dst_chw, 
                        uint16_t W, 
                        uint16_t C) {
  const uint32_t plane = (uint32_t)W * W;

  for (uint16_t y = 0; y < W; y++) {
    for (uint16_t x = 0; x < W; x++) {
      const uint32_t pix = (uint32_t)y * W + x;
      for (uint16_t c = 0; c < C; c++) {
        dst_chw[(uint32_t)c * plane + pix] = src_hwc[pix * C + c];
      }
    }
  }
  return (W * W * C);
}


uint16_t noodle_gap(float *inout, uint16_t C, uint16_t W) {
  const uint16_t n = (uint16_t)(W * W);

  // Always safe in-place if we write outputs from high->low channel index.
  // This avoids needing temp_buff1 even when n < C.
  for (int32_t c = (int32_t)C - 1; c >= 0; --c) {
    const float *plane = inout + (uint32_t)c * n;
    double acc = 0.0;
    for (uint16_t i = 0; i < n; i++) acc += (double)plane[i];
    inout[c] = (float)(acc / (double)n);
  }
  return C;
}

uint16_t noodle_gmp(float *inout,
                    uint16_t C,
                    uint16_t W)
{
  for (uint16_t c = 0; c < C; c++) {
    const float *in_ch = inout + (size_t)c * W;
    float v = in_ch[0];

    for (uint16_t i = 1; i < W; i++){
      if (in_ch[i] > v)
        v = in_ch[i];
    }

    inout[c] = v;
  }

  return C;
}

// ===== NoodleBuffer convenience wrappers =====



uint16_t noodle_flat(const char *in_fn,
                     NoodleBuffer *output,
                     uint16_t V,
                     uint16_t n_filters) {
  if (!in_fn || !output) return 0;

  const size_t n = (size_t)V * (size_t)V * (size_t)n_filters;
  float *out = noodle_buffer_require(output, n);
  if (!out) return 0;

  return noodle_flat(in_fn, out, V, n_filters);
}

uint16_t noodle_flat(NoodleBuffer *input,
                     NoodleBuffer *output,
                     uint16_t V,
                     uint16_t n_filters) {
  if (!input || !input->data || !output) return 0;

  const size_t n = (size_t)V * (size_t)V * (size_t)n_filters;
  float *out = noodle_buffer_require(output, n);
  if (!out) return 0;

  return noodle_flat(input->data, out, V, n_filters);
}

uint16_t noodle_reshape(NoodleBuffer *src_hwc,
                        NoodleBuffer *dst_chw,
                        uint16_t W,
                        uint16_t C) {
  if (!src_hwc || !src_hwc->data || !dst_chw) return 0;

  const size_t n = (size_t)W * (size_t)W * (size_t)C;
  float *dst = noodle_buffer_require(dst_chw, n);
  if (!dst) return 0;

  return noodle_reshape(src_hwc->data, dst, W, C);
}

uint16_t noodle_gap(NoodleBuffer *inout, uint16_t C, uint16_t W) {
  if (!inout || !inout->data) return 0;
  return noodle_gap(inout->data, C, W);
}

uint16_t noodle_gmp(NoodleBuffer *inout, uint16_t C, uint16_t W) {
  if (!inout || !inout->data) return 0;
  return noodle_gmp(inout->data, C, W);
}

uint16_t noodle_concat(NoodleBuffer *A,
                       uint16_t C_A,
                       NoodleBuffer *B,
                       uint16_t C_B,
                       NoodleBuffer *output,
                       uint16_t V)
{
  if (!A || !B || !output) return 0;
  if (!A->data || !B->data) return 0;

  const uint32_t plane = (uint32_t)V * (uint32_t)V;
  const uint16_t C_out = C_A + C_B;
  const uint32_t total = (uint32_t)C_out * plane;

  float *Y = noodle_buffer_require(output, total);
  if (!Y) return 0;

  // Copy A channels first.
  const uint32_t nA = (uint32_t)C_A * plane;
  for (uint32_t i = 0; i < nA; i++) {
    Y[i] = A->data[i];
  }

  // Copy B channels after A.
  const uint32_t nB = (uint32_t)C_B * plane;
  const uint32_t offset = nA;
  for (uint32_t i = 0; i < nB; i++) {
    Y[offset + i] = B->data[i];
  }

  return C_out;
}

uint16_t noodle_pool2d(NoodleBuffer *input,
                       uint16_t C,
                       uint16_t W,
                       NoodleBuffer *output,
                       uint16_t K,
                       uint16_t S) {
  if (!input || !output) return 0;
  if (!input->data) return 0;
  if (C == 0 || W == 0 || K == 0 || S == 0) return 0;

  // Keep this wrapper out-of-place for safety.
  // In-place pooling can overwrite values that are still needed,
  // especially for overlapping pooling where S < K.
  if (input == output) return 0;

#if NOODLE_POOL_MODE == NOODLE_POOL_NONE
  const uint16_t Wo = W;
#else
  const uint16_t Wo = (K == 1 && S == 1)
                    ? W
                    : (uint16_t)((W - K) / S + 1);
#endif

  const uint32_t in_plane  = (uint32_t)W  * (uint32_t)W;
  const uint32_t out_plane = (uint32_t)Wo * (uint32_t)Wo;

  float *Y = noodle_buffer_require(output, (size_t)C * out_plane);
  if (!Y) return 0;

  for (uint16_t c = 0; c < C; c++) {
    const float *x_c = input->data + (uint32_t)c * in_plane;
    float *y_c = Y + (uint32_t)c * out_plane;

    const uint16_t got = noodle_do_pooling(x_c, W, K, S, y_c);
    if (got != Wo) return 0;
  }

  return Wo;
}