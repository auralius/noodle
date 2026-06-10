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
