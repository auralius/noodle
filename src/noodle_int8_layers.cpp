#include "noodle_config.h"
#if defined(NOODLE_USE_INT8)
#include "noodle_int8.h"
#include "noodle_internal.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int32_t *g_q_acc = nullptr;
static size_t g_q_acc_capacity = 0;
static int8_t *g_q_filter = nullptr;
static size_t g_q_filter_capacity = 0;

#if defined(ARDUINO_ARCH_ESP32)
#include <Arduino.h>
#endif

static void *q_alloc(size_t bytes) {
  if (!bytes) return nullptr;
  void *p = malloc(bytes);
#if defined(ARDUINO_ARCH_ESP32)
  if (!p && psramFound()) p = ps_malloc(bytes);
#endif
  return p;
}

static int32_t *q_acc_require(size_t n) {
  if (n <= g_q_acc_capacity) return g_q_acc;
  if (n > static_cast<size_t>(-1) / sizeof(int32_t)) return nullptr;
  int32_t *p = static_cast<int32_t *>(q_alloc(n * sizeof(int32_t)));
  if (!p) return nullptr;
  free(g_q_acc);
  g_q_acc = p;
  g_q_acc_capacity = n;
  return p;
}

static int8_t *q_filter_require(size_t n) {
  if (n <= g_q_filter_capacity) return g_q_filter;
  int8_t *p = static_cast<int8_t *>(q_alloc(n));
  if (!p) return nullptr;
  free(g_q_filter);
  g_q_filter = p;
  g_q_filter_capacity = n;
  return p;
}

void noodle_i8_temp_buffers_free(void) {
  free(g_q_acc);
  free(g_q_filter);
  g_q_acc = nullptr;
  g_q_filter = nullptr;
  g_q_acc_capacity = 0;
  g_q_filter_capacity = 0;
}

static int32_t q_sat_i32(int64_t x) {
  if (x > INT32_MAX) return INT32_MAX;
  if (x < INT32_MIN) return INT32_MIN;
  return static_cast<int32_t>(x);
}

static bool q_scale_equal(float a, float b) {
  const float d = fabsf(a - b);
  const float m = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
  return d <= 1e-6f * m;
}

static bool q_input_matches(const NoodleTensor *input,
                            float scale, int32_t zp) {
  return input && q_scale_equal(input->scale, scale) && input->zero_point == zp;
}

static int32_t q_apply_output(int32_t acc, int32_t multiplier, int32_t shift,
                              int32_t output_zp, int32_t act_min,
                              int32_t act_max) {
  int32_t q = noodle_i8_multiply_by_quantized_multiplier(acc, multiplier, shift);
  int64_t with_zp = static_cast<int64_t>(q) + output_zp;
  q = q_sat_i32(with_zp);
  if (q < act_min) q = act_min;
  if (q > act_max) q = act_max;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return q;
}

struct QGeom {
  uint16_t V;
  uint16_t P0;
  uint16_t P1;
};

static bool q_conv_geom(uint16_t W, uint16_t K, uint16_t P, uint16_t S,
                        QGeom *g) {
  if (!g || !W || !K || !S) return false;
  if (P == NOODLE_SAME_PADDING) {
    const uint32_t V = (static_cast<uint32_t>(W) + S - 1u) / S;
    const uint32_t needed = V ? (V - 1u) * S + K : 0;
    const uint32_t total = needed > W ? needed - W : 0;
    if (V > 65535u || total > 131070u) return false;
    g->V = static_cast<uint16_t>(V);
    g->P0 = static_cast<uint16_t>(total / 2u);
    g->P1 = static_cast<uint16_t>(total - total / 2u);
    return true;
  }
  const int32_t num = static_cast<int32_t>(W) - K + 2 * static_cast<int32_t>(P);
  if (num < 0) return false;
  g->V = static_cast<uint16_t>(num / S + 1);
  g->P0 = g->P1 = P;
  return g->V != 0;
}

static size_t q_read_i8_block(NDL_File &f, int8_t *dst, size_t n) {
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  return noodle_read_raw(f, dst, n);
#else
  for (size_t i = 0; i < n; ++i) dst[i] = noodle_read_q8(f);
  return n;
#endif
}

static bool q_open_required(const char *fn, NDL_File *f) {
  if (!fn || !f) return false;
  *f = noodle_fs_open_read(fn);
  return static_cast<bool>(*f);
}

static void q_close_if(NDL_File *f) {
  if (f && static_cast<bool>(*f)) f->close();
}

static int32_t q_conv2d_at(const int8_t *input, uint16_t I, uint16_t W,
                           const int8_t *filter, uint16_t K,
                           uint16_t oy, uint16_t ox,
                           uint16_t P0, uint16_t S,
                           int32_t input_zp, int32_t bias) {
  int64_t acc = bias;
  const uint32_t in_plane = static_cast<uint32_t>(W) * W;
  const uint32_t kernel_plane = static_cast<uint32_t>(K) * K;
  for (uint16_t ic = 0; ic < I; ++ic) {
    const int8_t *x = input + static_cast<uint32_t>(ic) * in_plane;
    const int8_t *w = filter + static_cast<uint32_t>(ic) * kernel_plane;
    for (uint16_t ky = 0; ky < K; ++ky) {
      const int32_t iy = static_cast<int32_t>(oy) * S + ky - P0;
      if (iy < 0 || iy >= W) continue;
      for (uint16_t kx = 0; kx < K; ++kx) {
        const int32_t ix = static_cast<int32_t>(ox) * S + kx - P0;
        if (ix < 0 || ix >= W) continue;
        const int32_t xv = static_cast<int32_t>(x[static_cast<uint32_t>(iy) * W + ix]) - input_zp;
        acc += static_cast<int64_t>(xv) * w[static_cast<uint32_t>(ky) * K + kx];
      }
    }
  }
  return q_sat_i32(acc);
}

static bool q_conv2d_filter(const NoodleTensor *input, uint16_t V,
                            uint16_t P0, const int8_t *filter,
                            int32_t bias, int32_t multiplier, int32_t shift,
                            uint16_t K, uint16_t S,
                            int32_t input_zp, int32_t output_zp,
                            int32_t act_min, int32_t act_max,
                            int8_t *out_plane) {
  if (!input || !filter || !out_plane) return false;
  const int8_t *src = input->buffer.data;
  for (uint16_t oy = 0; oy < V; ++oy)
    for (uint16_t ox = 0; ox < V; ++ox) {
      const int32_t acc = q_conv2d_at(src, input->C, input->W, filter, K,
                                      oy, ox, P0, S, input_zp, bias);
      out_plane[static_cast<uint32_t>(oy) * V + ox] = static_cast<int8_t>(
          q_apply_output(acc, multiplier, shift, output_zp, act_min, act_max));
    }
  return true;
}

uint16_t noodle_i8_conv2d(NoodleTensor *input, const ConvMem &layer,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  const uint32_t filter_size = static_cast<uint32_t>(input->C) * layer.K * layer.K;
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    const int8_t *filter = layer.weight + static_cast<uint32_t>(oc) * filter_size;
    const int32_t bias = layer.bias ? layer.bias[oc] : 0;
    if (!q_conv2d_filter(input, g.V, g.P0, filter, bias,
                         layer.multiplier[oc], layer.shift[oc],
                         layer.K, layer.S, layer.input_zero_point,
                         layer.output_zero_point, layer.activation_min,
                         layer.activation_max,
                         dst + static_cast<uint32_t>(oc) * out_plane)) return 0;
  }
  return g.V;
}

uint16_t noodle_i8_conv2d(NoodleTensor *input, const Conv &layer,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;

  NDL_File wf, bf, mf, sf;
  if (!q_open_required(layer.weight_fn, &wf) ||
      !q_open_required(layer.multiplier_fn, &mf) ||
      !q_open_required(layer.shift_fn, &sf)) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const bool has_bias = layer.bias_fn && q_open_required(layer.bias_fn, &bf);
  if (layer.bias_fn && !has_bias) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }

  const size_t filter_size = static_cast<size_t>(input->C) * layer.K * layer.K;
  int8_t *filter = q_filter_require(filter_size);
  if (!filter) {
    q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    if (q_read_i8_block(wf, filter, filter_size) != filter_size) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    const int32_t bias = has_bias ? noodle_read_i32(bf) : 0;
    const int32_t mult = noodle_read_i32(mf);
    const int32_t shift = noodle_read_i32(sf);
    if (!q_conv2d_filter(input, g.V, g.P0, filter, bias, mult, shift,
                         layer.K, layer.S, layer.input_zero_point,
                         layer.output_zero_point, layer.activation_min,
                         layer.activation_max,
                         dst + static_cast<uint32_t>(oc) * out_plane)) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
  }
  q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
  return g.V;
}

static int32_t q_conv1d_at(const int8_t *input, uint16_t I, uint16_t W,
                           const int8_t *filter, uint16_t K, uint16_t o,
                           uint16_t P, uint16_t S, int32_t input_zp,
                           int32_t bias) {
  int64_t acc = bias;
  for (uint16_t ic = 0; ic < I; ++ic) {
    const int8_t *x = input + static_cast<size_t>(ic) * W;
    const int8_t *w = filter + static_cast<size_t>(ic) * K;
    for (uint16_t k = 0; k < K; ++k) {
      const int32_t ix = static_cast<int32_t>(o) * S + k - P;
      if (ix < 0 || ix >= W) continue;
      acc += static_cast<int64_t>(static_cast<int32_t>(x[ix]) - input_zp) * w[k];
    }
  }
  return q_sat_i32(acc);
}

uint16_t noodle_i8_conv1d(NoodleTensor *input, const ConvMem &layer,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_1D || !layer.weight ||
      !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_1d(output, layer.O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    const int8_t *filter = layer.weight + static_cast<size_t>(oc) * filter_size;
    const int32_t bias = layer.bias ? layer.bias[oc] : 0;
    for (uint16_t o = 0; o < g.V; ++o) {
      const int32_t acc = q_conv1d_at(input->buffer.data, input->C, input->W,
                                      filter, layer.K, o, g.P0, layer.S,
                                      layer.input_zero_point, bias);
      dst[static_cast<size_t>(oc) * g.V + o] = static_cast<int8_t>(
          q_apply_output(acc, layer.multiplier[oc], layer.shift[oc],
                         layer.output_zero_point, layer.activation_min,
                         layer.activation_max));
    }
  }
  return g.V;
}

uint16_t noodle_i8_conv1d(NoodleTensor *input, const Conv &layer,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_1D || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_1d(output, layer.O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  NDL_File wf, bf, mf, sf;
  if (!q_open_required(layer.weight_fn, &wf) ||
      !q_open_required(layer.multiplier_fn, &mf) ||
      !q_open_required(layer.shift_fn, &sf)) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const bool has_bias = layer.bias_fn && q_open_required(layer.bias_fn, &bf);
  if (layer.bias_fn && !has_bias) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K;
  int8_t *filter = q_filter_require(filter_size);
  if (!filter) { q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0; }
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    if (q_read_i8_block(wf, filter, filter_size) != filter_size) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    const int32_t bias = has_bias ? noodle_read_i32(bf) : 0;
    const int32_t mult = noodle_read_i32(mf);
    const int32_t shift = noodle_read_i32(sf);
    for (uint16_t o = 0; o < g.V; ++o) {
      const int32_t acc = q_conv1d_at(input->buffer.data, input->C, input->W,
                                      filter, layer.K, o, g.P0, layer.S,
                                      layer.input_zero_point, bias);
      dst[static_cast<size_t>(oc) * g.V + o] = static_cast<int8_t>(
          q_apply_output(acc, mult, shift, layer.output_zero_point,
                         layer.activation_min, layer.activation_max));
    }
  }
  q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
  return g.V;
}

static int32_t q_dw_at(const NoodleTensor *input, uint16_t ic,
                       const int8_t *kernel, uint16_t K,
                       uint16_t oy, uint16_t ox, uint16_t P0, uint16_t S,
                       int32_t input_zp, int32_t bias) {
  const uint32_t plane = static_cast<uint32_t>(input->W) * input->W;
  const int8_t *x = input->buffer.data + static_cast<uint32_t>(ic) * plane;
  int64_t acc = bias;
  for (uint16_t ky = 0; ky < K; ++ky) {
    const int32_t iy = static_cast<int32_t>(oy) * S + ky - P0;
    if (iy < 0 || iy >= input->W) continue;
    for (uint16_t kx = 0; kx < K; ++kx) {
      const int32_t ix = static_cast<int32_t>(ox) * S + kx - P0;
      if (ix < 0 || ix >= input->W) continue;
      const int32_t xv = static_cast<int32_t>(x[static_cast<uint32_t>(iy) * input->W + ix]) - input_zp;
      acc += static_cast<int64_t>(xv) * kernel[static_cast<uint32_t>(ky) * K + kx];
    }
  }
  return q_sat_i32(acc);
}

uint16_t noodle_i8_dwconv2d(NoodleTensor *input, const ConvMem &layer,
                          NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || layer.depth_multiplier == 0 ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint32_t expected_o = static_cast<uint32_t>(input->C) * layer.depth_multiplier;
  const uint16_t O = layer.O ? layer.O : static_cast<uint16_t>(expected_o);
  if (expected_o > 65535u || O != expected_o) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  const uint32_t ksize = static_cast<uint32_t>(layer.K) * layer.K;
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < O; ++oc) {
    const uint16_t ic = oc / layer.depth_multiplier;
    const int8_t *kernel = layer.weight + static_cast<uint32_t>(oc) * ksize;
    const int32_t bias = layer.bias ? layer.bias[oc] : 0;
    int8_t *out = dst + static_cast<uint32_t>(oc) * out_plane;
    for (uint16_t oy = 0; oy < g.V; ++oy)
      for (uint16_t ox = 0; ox < g.V; ++ox) {
        const int32_t acc = q_dw_at(input, ic, kernel, layer.K, oy, ox,
                                    g.P0, layer.S, layer.input_zero_point, bias);
        out[static_cast<uint32_t>(oy) * g.V + ox] = static_cast<int8_t>(
            q_apply_output(acc, layer.multiplier[oc], layer.shift[oc],
                           layer.output_zero_point, layer.activation_min,
                           layer.activation_max));
      }
  }
  return g.V;
}

uint16_t noodle_i8_dwconv2d(NoodleTensor *input, const Conv &layer,
                          NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || layer.depth_multiplier == 0 ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint32_t expected_o = static_cast<uint32_t>(input->C) * layer.depth_multiplier;
  const uint16_t O = layer.O ? layer.O : static_cast<uint16_t>(expected_o);
  if (expected_o > 65535u || O != expected_o) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, O, g.V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  NDL_File wf, bf, mf, sf;
  if (!q_open_required(layer.weight_fn, &wf) ||
      !q_open_required(layer.multiplier_fn, &mf) ||
      !q_open_required(layer.shift_fn, &sf)) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const bool has_bias = layer.bias_fn && q_open_required(layer.bias_fn, &bf);
  if (layer.bias_fn && !has_bias) { q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0; }
  const size_t ksize = static_cast<size_t>(layer.K) * layer.K;
  int8_t *kernel = q_filter_require(ksize);
  if (!kernel) { q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0; }
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < O; ++oc) {
    if (q_read_i8_block(wf, kernel, ksize) != ksize) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    const int32_t bias = has_bias ? noodle_read_i32(bf) : 0;
    const int32_t mult = noodle_read_i32(mf);
    const int32_t shift = noodle_read_i32(sf);
    const uint16_t ic = oc / layer.depth_multiplier;
    int8_t *out = dst + static_cast<uint32_t>(oc) * out_plane;
    for (uint16_t oy = 0; oy < g.V; ++oy)
      for (uint16_t ox = 0; ox < g.V; ++ox) {
        const int32_t acc = q_dw_at(input, ic, kernel, layer.K, oy, ox,
                                    g.P0, layer.S, layer.input_zero_point, bias);
        out[static_cast<uint32_t>(oy) * g.V + ox] = static_cast<int8_t>(
            q_apply_output(acc, mult, shift, layer.output_zero_point,
                           layer.activation_min, layer.activation_max));
      }
  }
  q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
  return g.V;
}

static uint16_t q_transpose_width(uint16_t W, uint16_t K, uint16_t P,
                                  uint16_t S, uint16_t OP) {
  if (!W || !K || !S || P == NOODLE_SAME_PADDING) return 0;
  const int64_t v = (static_cast<int64_t>(W) - 1) * S - 2 * P + K + OP;
  return v > 0 && v <= 65535 ? static_cast<uint16_t>(v) : 0;
}

static bool q_transpose_filter(const NoodleTensor *input, uint16_t V,
                               const int8_t *filter, uint16_t K,
                               uint16_t P, uint16_t S, int32_t input_zp,
                               int32_t bias, int32_t mult, int32_t shift,
                               int32_t output_zp, int32_t act_min,
                               int32_t act_max, int8_t *out) {
  const uint32_t plane = static_cast<uint32_t>(V) * V;
  int32_t *acc = q_acc_require(plane);
  if (!acc) return false;
  for (uint32_t i = 0; i < plane; ++i) acc[i] = bias;
  const uint32_t in_plane = static_cast<uint32_t>(input->W) * input->W;
  const uint32_t kplane = static_cast<uint32_t>(K) * K;
  for (uint16_t ic = 0; ic < input->C; ++ic) {
    const int8_t *x = input->buffer.data + static_cast<uint32_t>(ic) * in_plane;
    const int8_t *w = filter + static_cast<uint32_t>(ic) * kplane;
    for (uint16_t iy = 0; iy < input->W; ++iy)
      for (uint16_t ix = 0; ix < input->W; ++ix) {
        const int32_t xv = static_cast<int32_t>(x[static_cast<uint32_t>(iy) * input->W + ix]) - input_zp;
        for (uint16_t ky = 0; ky < K; ++ky) {
          const int32_t oy = static_cast<int32_t>(iy) * S + ky - P;
          if (oy < 0 || oy >= V) continue;
          for (uint16_t kx = 0; kx < K; ++kx) {
            const int32_t ox = static_cast<int32_t>(ix) * S + kx - P;
            if (ox < 0 || ox >= V) continue;
            const uint32_t oi = static_cast<uint32_t>(oy) * V + ox;
            const int64_t next = static_cast<int64_t>(acc[oi]) +
                static_cast<int64_t>(xv) * w[static_cast<uint32_t>(ky) * K + kx];
            acc[oi] = q_sat_i32(next);
          }
        }
      }
  }
  for (uint32_t i = 0; i < plane; ++i)
    out[i] = static_cast<int8_t>(q_apply_output(acc[i], mult, shift,
                                                output_zp, act_min, act_max));
  return true;
}

uint16_t noodle_i8_conv_transpose2d(NoodleTensor *input,
                                  const ConvMem &layer,
                                  NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint16_t V = q_transpose_width(input->W, layer.K, layer.P, layer.S, layer.OP);
  if (!V) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K * layer.K;
  const uint32_t plane = static_cast<uint32_t>(V) * V;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    if (!q_transpose_filter(input, V, layer.weight + static_cast<size_t>(oc) * filter_size,
                            layer.K, layer.P, layer.S, layer.input_zero_point,
                            layer.bias ? layer.bias[oc] : 0,
                            layer.multiplier[oc], layer.shift[oc],
                            layer.output_zero_point, layer.activation_min,
                            layer.activation_max,
                            dst + static_cast<uint32_t>(oc) * plane)) return 0;
  }
  return V;
}

uint16_t noodle_i8_conv_transpose2d(NoodleTensor *input,
                                  const Conv &layer,
                                  NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint16_t V = q_transpose_width(input->W, layer.K, layer.P, layer.S, layer.OP);
  if (!V) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, V,
                                          layer.output_scale,
                                          layer.output_zero_point);
  if (!dst) return 0;
  NDL_File wf, bf, mf, sf;
  if (!q_open_required(layer.weight_fn, &wf) ||
      !q_open_required(layer.multiplier_fn, &mf) ||
      !q_open_required(layer.shift_fn, &sf)) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const bool has_bias = layer.bias_fn && q_open_required(layer.bias_fn, &bf);
  if (layer.bias_fn && !has_bias) { q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0; }
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K * layer.K;
  int8_t *filter = q_filter_require(filter_size);
  if (!filter) { q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0; }
  const uint32_t plane = static_cast<uint32_t>(V) * V;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    if (q_read_i8_block(wf, filter, filter_size) != filter_size) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    const int32_t bias = has_bias ? noodle_read_i32(bf) : 0;
    const int32_t mult = noodle_read_i32(mf);
    const int32_t shift = noodle_read_i32(sf);
    if (!q_transpose_filter(input, V, filter, layer.K, layer.P, layer.S,
                            layer.input_zero_point, bias, mult, shift,
                            layer.output_zero_point, layer.activation_min,
                            layer.activation_max,
                            dst + static_cast<uint32_t>(oc) * plane)) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
  }
  q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
  return V;
}

static int32_t q_fcn_dot(const int8_t *input, const int8_t *weight,
                         size_t I, int32_t input_zp, int32_t bias) {
  int64_t acc = bias;
  for (size_t i = 0; i < I; ++i)
    acc += static_cast<int64_t>(static_cast<int32_t>(input[i]) - input_zp) * weight[i];
  return q_sat_i32(acc);
}

uint16_t noodle_i8_fcn(NoodleTensor *input, const FCNMem &layer,
                     NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      !layer.weight || !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const size_t I = noodle_tensor_size(input);
  int8_t *dst = noodle_i8_tensor_require_vector(output, layer.O,
                                               layer.output_scale,
                                               layer.output_zero_point);
  if (!dst) return 0;
  const int8_t *src = input->buffer.data;
  for (uint16_t o = 0; o < layer.O; ++o) {
    const int32_t acc = q_fcn_dot(src, layer.weight + static_cast<size_t>(o) * I,
                                  I, layer.input_zero_point,
                                  layer.bias ? layer.bias[o] : 0);
    dst[o] = static_cast<int8_t>(q_apply_output(acc, layer.multiplier[o],
                                                layer.shift[o],
                                                layer.output_zero_point,
                                                layer.activation_min,
                                                layer.activation_max));
  }
  return layer.O;
}

uint16_t noodle_i8_fcn(NoodleTensor *input, const FCNFile &layer,
                     NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      !layer.O || !q_input_matches(input, layer.input_scale,
                                    layer.input_zero_point)) return 0;
  const size_t I = noodle_tensor_size(input);
  int8_t *dst = noodle_i8_tensor_require_vector(output, layer.O,
                                               layer.output_scale,
                                               layer.output_zero_point);
  if (!dst) return 0;
  NDL_File wf, bf, mf, sf;
  if (!q_open_required(layer.weight_fn, &wf) ||
      !q_open_required(layer.multiplier_fn, &mf) ||
      !q_open_required(layer.shift_fn, &sf)) {
    q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0;
  }
  const bool has_bias = layer.bias_fn && q_open_required(layer.bias_fn, &bf);
  if (layer.bias_fn && !has_bias) { q_close_if(&wf); q_close_if(&mf); q_close_if(&sf); return 0; }
  int8_t *row = q_filter_require(I);
  if (!row) { q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0; }
  const int8_t *src = input->buffer.data;
  for (uint16_t o = 0; o < layer.O; ++o) {
    if (q_read_i8_block(wf, row, I) != I) {
      q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf); return 0;
    }
    const int32_t acc = q_fcn_dot(src, row, I, layer.input_zero_point,
                                  has_bias ? noodle_read_i32(bf) : 0);
    const int32_t mult = noodle_read_i32(mf);
    const int32_t shift = noodle_read_i32(sf);
    dst[o] = static_cast<int8_t>(q_apply_output(acc, mult, shift,
                                                layer.output_zero_point,
                                                layer.activation_min,
                                                layer.activation_max));
  }
  q_close_if(&wf); q_close_if(&bf); q_close_if(&mf); q_close_if(&sf);
  return layer.O;
}


uint16_t noodle_i8_conv2d(NoodleTensor *input, const ConvProgmem &layer,
                          NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || !layer.O ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, layer.O, g.V,
                                            layer.output_scale,
                                            layer.output_zero_point);
  if (!dst) return 0;
  const size_t filter_size = static_cast<size_t>(input->C) * layer.K * layer.K;
  int8_t *filter = q_filter_require(filter_size);
  if (!filter) return 0;
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < layer.O; ++oc) {
    const uint32_t base = static_cast<uint32_t>(oc) * filter_size;
    for (size_t i = 0; i < filter_size; ++i) {
      filter[i] = noodle_pgm_i8(layer.weight, base + static_cast<uint32_t>(i));
    }
    const int32_t bias = layer.bias ? noodle_pgm_i32(layer.bias, oc) : 0;
    const int32_t mult = noodle_pgm_i32(layer.multiplier, oc);
    const int32_t shift = noodle_pgm_i32(layer.shift, oc);
    if (!q_conv2d_filter(input, g.V, g.P0, filter, bias, mult, shift,
                         layer.K, layer.S, layer.input_zero_point,
                         layer.output_zero_point, layer.activation_min,
                         layer.activation_max,
                         dst + static_cast<uint32_t>(oc) * out_plane)) return 0;
  }
  return g.V;
}

uint16_t noodle_i8_dwconv2d(NoodleTensor *input, const ConvProgmem &layer,
                            NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || !layer.weight ||
      !layer.multiplier || !layer.shift || layer.depth_multiplier == 0 ||
      !q_input_matches(input, layer.input_scale, layer.input_zero_point)) return 0;
  const uint32_t expected_o = static_cast<uint32_t>(input->C) * layer.depth_multiplier;
  const uint16_t O = layer.O ? layer.O : static_cast<uint16_t>(expected_o);
  if (expected_o > 65535u || O != expected_o) return 0;
  QGeom g;
  if (!q_conv_geom(input->W, layer.K, layer.P, layer.S, &g)) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, O, g.V,
                                            layer.output_scale,
                                            layer.output_zero_point);
  if (!dst) return 0;
  const size_t ksize = static_cast<size_t>(layer.K) * layer.K;
  int8_t *kernel = q_filter_require(ksize);
  if (!kernel) return 0;
  const uint32_t out_plane = static_cast<uint32_t>(g.V) * g.V;
  for (uint16_t oc = 0; oc < O; ++oc) {
    const uint32_t base = static_cast<uint32_t>(oc) * ksize;
    for (size_t i = 0; i < ksize; ++i) {
      kernel[i] = noodle_pgm_i8(layer.weight, base + static_cast<uint32_t>(i));
    }
    const uint16_t ic = oc / layer.depth_multiplier;
    const int32_t bias = layer.bias ? noodle_pgm_i32(layer.bias, oc) : 0;
    const int32_t mult = noodle_pgm_i32(layer.multiplier, oc);
    const int32_t shift = noodle_pgm_i32(layer.shift, oc);
    int8_t *out = dst + static_cast<uint32_t>(oc) * out_plane;
    for (uint16_t oy = 0; oy < g.V; ++oy) {
      for (uint16_t ox = 0; ox < g.V; ++ox) {
        const int32_t acc = q_dw_at(input, ic, kernel, layer.K, oy, ox,
                                    g.P0, layer.S, layer.input_zero_point, bias);
        out[static_cast<uint32_t>(oy) * g.V + ox] = static_cast<int8_t>(
            q_apply_output(acc, mult, shift, layer.output_zero_point,
                           layer.activation_min, layer.activation_max));
      }
    }
  }
  return g.V;
}

uint16_t noodle_i8_fcn(NoodleTensor *input, const FCNProgmem &layer,
                       NoodleTensor *output) {
#if defined(__AVR__)
  if (!input || !output || input == output || !input->buffer.data ||
      !layer.weight_far || !layer.multiplier_far || !layer.shift_far ||
      !layer.O || !q_input_matches(input, layer.input_scale,
                                    layer.input_zero_point)) return 0;
  const size_t I = noodle_tensor_size(input);
  int8_t *dst = noodle_i8_tensor_require_vector(output, layer.O,
                                                 layer.output_scale,
                                                 layer.output_zero_point);
  if (!dst) return 0;
  int8_t *row = q_filter_require(I);
  if (!row) return 0;
  for (uint16_t o = 0; o < layer.O; ++o) {
    const uint32_t row_addr = layer.weight_far + static_cast<uint32_t>(o * I);
    for (size_t i = 0; i < I; ++i) {
      row[i] = noodle_pgm_i8_address(row_addr + static_cast<uint32_t>(i));
    }
    const int32_t bias = layer.bias_far
        ? noodle_pgm_i32_address(layer.bias_far + static_cast<uint32_t>(o) * 4u)
        : 0;
    const int32_t mult = noodle_pgm_i32_address(
        layer.multiplier_far + static_cast<uint32_t>(o) * 4u);
    const int32_t shift = noodle_pgm_i32_address(
        layer.shift_far + static_cast<uint32_t>(o) * 4u);
    const int32_t acc = q_fcn_dot(input->buffer.data, row, I,
                                  layer.input_zero_point, bias);
    dst[o] = static_cast<int8_t>(q_apply_output(
        acc, mult, shift, layer.output_zero_point,
        layer.activation_min, layer.activation_max));
  }
  return layer.O;
#else
  (void)input; (void)layer; (void)output;
  return 0;
#endif
}

#endif  // NOODLE_USE_INT8
