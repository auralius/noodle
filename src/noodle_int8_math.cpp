#include "noodle_config.h"
#if defined(NOODLE_USE_INT8)
#include "noodle_int8.h"
#include "noodle_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int32_t q_clamp_i32(int64_t x) {
  if (x > INT32_MAX) return INT32_MAX;
  if (x < INT32_MIN) return INT32_MIN;
  return static_cast<int32_t>(x);
}

int8_t noodle_i8_clamp(int32_t x) {
  if (x > 127) return 127;
  if (x < -128) return -128;
  return static_cast<int8_t>(x);
}

int32_t noodle_i8_saturating_rounding_doubling_high_mul(int32_t a, int32_t b) {
  if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
  const int64_t ab = static_cast<int64_t>(a) * static_cast<int64_t>(b);
  const int64_t nudge = ab >= 0 ? (INT64_C(1) << 30)
                                : (INT64_C(1) - (INT64_C(1) << 30));
  return static_cast<int32_t>((ab + nudge) / (INT64_C(1) << 31));
}

int32_t noodle_i8_rounding_divide_by_pot(int32_t x, int exponent) {
  if (exponent <= 0) return x;
  if (exponent > 31) return x < 0 ? -1 : 0;
  const uint32_t mask = (UINT32_C(1) << exponent) - 1u;
  const uint32_t remainder = static_cast<uint32_t>(x) & mask;
  const uint32_t threshold = (mask >> 1) + (x < 0 ? 1u : 0u);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

int32_t noodle_i8_multiply_by_quantized_multiplier(int32_t x,
                                                 int32_t multiplier,
                                                 int32_t shift) {
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  int64_t shifted = static_cast<int64_t>(x);
  if (left_shift >= 31) shifted = shifted >= 0 ? INT32_MAX : INT32_MIN;
  else shifted *= (INT64_C(1) << left_shift);
  const int32_t shifted32 = q_clamp_i32(shifted);
  const int32_t high = noodle_i8_saturating_rounding_doubling_high_mul(
      shifted32, multiplier);
  return noodle_i8_rounding_divide_by_pot(high, right_shift);
}

bool noodle_quantize_multiplier(double real_multiplier,
                                 int32_t *multiplier,
                                 int32_t *shift) {
  if (!multiplier || !shift || real_multiplier < 0.0 || !isfinite(real_multiplier)) {
    return false;
  }
  if (real_multiplier == 0.0) {
    *multiplier = 0;
    *shift = 0;
    return true;
  }
  int exp = 0;
  const double q = frexp(real_multiplier, &exp);
  int64_t q_fixed = static_cast<int64_t>(llround(q * (INT64_C(1) << 31)));
  if (q_fixed == (INT64_C(1) << 31)) {
    q_fixed >>= 1;
    ++exp;
  }
  if (exp < -31) {
    *multiplier = 0;
    *shift = 0;
    return true;
  }
  if (exp > 30) return false;
  *multiplier = static_cast<int32_t>(q_fixed);
  *shift = exp;
  return true;
}

int32_t noodle_quantize_float(float value, float scale, int32_t zero_point) {
  if (!(scale > 0.0f)) return zero_point;
  const double q = nearbyint(static_cast<double>(value) / scale) + zero_point;
  if (q > 127.0) return 127;
  if (q < -128.0) return -128;
  return static_cast<int32_t>(q);
}

float noodle_dequantize_int8(int8_t value, float scale, int32_t zero_point) {
  return (static_cast<int32_t>(value) - zero_point) * scale;
}

void noodle_quantize_array(const float *src, int8_t *dst, size_t n,
                            float scale, int32_t zero_point) {
  if (!src || !dst) return;
  for (size_t i = 0; i < n; ++i) {
    dst[i] = static_cast<int8_t>(noodle_quantize_float(src[i], scale, zero_point));
  }
}

void noodle_dequantize_array(const int8_t *src, float *dst, size_t n,
                              float scale, int32_t zero_point) {
  if (!src || !dst) return;
  for (size_t i = 0; i < n; ++i) {
    dst[i] = noodle_dequantize_int8(src[i], scale, zero_point);
  }
}

int32_t noodle_activation_min_relu(float output_scale,
                                    int32_t output_zero_point) {
  return noodle_quantize_float(0.0f, output_scale, output_zero_point);
}

int32_t noodle_read_i32(NDL_File &f) {
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  int32_t v = 0;
  return noodle_read_raw(f, &v, sizeof(v)) == sizeof(v) ? v : 0;
#else
  char s[24];
  size_t n = noodle_read_bytes_until(f, '\n', s, sizeof(s));
  s[n] = '\0';
  return static_cast<int32_t>(strtol(s, nullptr, 10));
#endif
}

void noodle_write_i32(NDL_File &f, int32_t value) {
#if NOODLE_FILE_FORMAT == NOODLE_FILE_FORMAT_BIN
  (void)noodle_write_raw(f, &value, sizeof(value));
#else
  char s[24];
  snprintf(s, sizeof(s), "%ld", static_cast<long>(value));
  for (const char *p = s; *p; ++p) f.write(static_cast<uint8_t>(*p));
  f.write(static_cast<uint8_t>('\n'));
#endif
}

void noodle_tensor_init(NoodleTensor *t) {
  if (!t) return;
  noodle_buffer_init(&t->buffer);
  t->C = t->W = 0;
  t->rank = NOODLE_TENSOR_EMPTY;
  t->scale = 1.0f;
  t->zero_point = 0;
}

void noodle_tensor_free(NoodleTensor *t) {
  if (!t) return;
  noodle_buffer_free(&t->buffer);
  t->C = t->W = 0;
  t->rank = NOODLE_TENSOR_EMPTY;
  t->scale = 1.0f;
  t->zero_point = 0;
}

static int8_t *q_tensor_require(NoodleTensor *t, uint16_t C, uint16_t W,
                                uint8_t rank, float scale, int32_t zp) {
  if (!t || C == 0 || W == 0 || !(scale > 0.0f) || zp < -128 || zp > 127) {
    return nullptr;
  }
  size_t n = static_cast<size_t>(C) * W;
  if (rank == NOODLE_TENSOR_2D) n *= W;
  int8_t *p = noodle_buffer_require(&t->buffer, n);
  if (!p) return nullptr;
  t->C = C;
  t->W = W;
  t->rank = rank;
  t->scale = scale;
  t->zero_point = zp;
  return p;
}

int8_t *noodle_i8_tensor_require_1d(NoodleTensor *t, uint16_t C, uint16_t W,
                                  float scale, int32_t zp) {
  return q_tensor_require(t, C, W, NOODLE_TENSOR_1D, scale, zp);
}
int8_t *noodle_i8_tensor_require_2d(NoodleTensor *t, uint16_t C, uint16_t W,
                                  float scale, int32_t zp) {
  return q_tensor_require(t, C, W, NOODLE_TENSOR_2D, scale, zp);
}
int8_t *noodle_i8_tensor_require_vector(NoodleTensor *t, uint16_t N,
                                      float scale, int32_t zp) {
  return q_tensor_require(t, N, 1, NOODLE_TENSOR_1D, scale, zp);
}

size_t noodle_tensor_size(const NoodleTensor *t) {
  if (!t) return 0;
  if (t->rank == NOODLE_TENSOR_1D) return static_cast<size_t>(t->C) * t->W;
  if (t->rank == NOODLE_TENSOR_2D) return static_cast<size_t>(t->C) * t->W * t->W;
  return 0;
}
size_t noodle_tensor_capacity(const NoodleTensor *t) {
  return t ? t->buffer.capacity : 0;
}
size_t noodle_tensor_capacity_bytes(const NoodleTensor *t) {
  return t ? t->buffer.capacity : 0;
}

static bool q_scale_equal(float a, float b) {
  const float d = fabsf(a - b);
  const float tol = 1e-7f * fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
  return d <= tol;
}

static bool q_same_params(const NoodleTensor *a, const NoodleTensor *b) {
  if (!a || !b) return false;
  return q_scale_equal(a->scale, b->scale) && a->zero_point == b->zero_point;
}

uint16_t noodle_i8_pool2d(NoodleTensor *input, const NoodleI8Pool &pool,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D || pool.M == 0 || pool.T == 0 ||
      input->W < pool.M) return 0;
  const uint16_t Wo = static_cast<uint16_t>((input->W - pool.M) / pool.T + 1);
  int8_t *dst = noodle_i8_tensor_require_2d(output, input->C, Wo,
                                          input->scale, input->zero_point);
  if (!dst) return 0;
  const uint32_t in_plane = static_cast<uint32_t>(input->W) * input->W;
  const uint32_t out_plane = static_cast<uint32_t>(Wo) * Wo;
  for (uint16_t c = 0; c < input->C; ++c) {
    const int8_t *src = input->buffer.data + static_cast<uint32_t>(c) * in_plane;
    int8_t *out = dst + static_cast<uint32_t>(c) * out_plane;
    for (uint16_t oy = 0; oy < Wo; ++oy) {
      for (uint16_t ox = 0; ox < Wo; ++ox) {
        const uint16_t y0 = oy * pool.T;
        const uint16_t x0 = ox * pool.T;
        if (pool.mode == NOODLE_I8_POOL_MAX) {
          int8_t v = src[static_cast<uint32_t>(y0) * input->W + x0];
          for (uint16_t ky = 0; ky < pool.M; ++ky)
            for (uint16_t kx = 0; kx < pool.M; ++kx) {
              const int8_t u = src[static_cast<uint32_t>(y0 + ky) * input->W + x0 + kx];
              if (u > v) v = u;
            }
          out[static_cast<uint32_t>(oy) * Wo + ox] = v;
        } else {
          int32_t sum = 0;
          for (uint16_t ky = 0; ky < pool.M; ++ky)
            for (uint16_t kx = 0; kx < pool.M; ++kx)
              sum += static_cast<int32_t>(src[static_cast<uint32_t>(y0 + ky) * input->W + x0 + kx]) - input->zero_point;
          const int32_t n = static_cast<int32_t>(pool.M) * pool.M;
          const int32_t rounded = sum >= 0 ? (sum + n / 2) / n : (sum - n / 2) / n;
          out[static_cast<uint32_t>(oy) * Wo + ox] = noodle_i8_clamp(rounded + input->zero_point);
        }
      }
    }
  }
  return Wo;
}

uint16_t noodle_i8_pool1d(NoodleTensor *input, const NoodleI8Pool &pool,
                        NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_1D || pool.M == 0 || pool.T == 0 ||
      input->W < pool.M) return 0;
  const uint16_t Wo = static_cast<uint16_t>((input->W - pool.M) / pool.T + 1);
  int8_t *dst = noodle_i8_tensor_require_1d(output, input->C, Wo,
                                          input->scale, input->zero_point);
  if (!dst) return 0;
  for (uint16_t c = 0; c < input->C; ++c) {
    const int8_t *src = input->buffer.data + static_cast<size_t>(c) * input->W;
    int8_t *out = dst + static_cast<size_t>(c) * Wo;
    for (uint16_t o = 0; o < Wo; ++o) {
      const uint16_t base = o * pool.T;
      if (pool.mode == NOODLE_I8_POOL_MAX) {
        int8_t v = src[base];
        for (uint16_t k = 1; k < pool.M; ++k) if (src[base + k] > v) v = src[base + k];
        out[o] = v;
      } else {
        int32_t sum = 0;
        for (uint16_t k = 0; k < pool.M; ++k) sum += static_cast<int32_t>(src[base + k]) - input->zero_point;
        const int32_t n = pool.M;
        const int32_t rounded = sum >= 0 ? (sum + n / 2) / n : (sum - n / 2) / n;
        out[o] = noodle_i8_clamp(rounded + input->zero_point);
      }
    }
  }
  return Wo;
}

uint16_t noodle_i8_gap(NoodleTensor *input, const NoodleI8Requant &rq,
                     NoodleTensor *output) {
  if (!input || !output || !input->buffer.data || input->rank != NOODLE_TENSOR_2D) return 0;
  const uint16_t C = input->C;
  const uint16_t W = input->W;
  const int32_t input_zp = input->zero_point;
  const uint32_t n = static_cast<uint32_t>(W) * W;
  int8_t *dst = noodle_i8_tensor_require_vector(output, C,
                                               rq.output_scale,
                                               rq.output_zero_point);
  if (!dst) return 0;
  // High-to-low channel order also makes the reduction safe when input==output.
  for (int32_t c = static_cast<int32_t>(C) - 1; c >= 0; --c) {
    const int8_t *src = input->buffer.data + static_cast<uint32_t>(c) * n;
    int32_t sum = 0;
    for (uint32_t i = 0; i < n; ++i) sum += static_cast<int32_t>(src[i]) - input_zp;
    int32_t q = noodle_i8_multiply_by_quantized_multiplier(sum, rq.multiplier, rq.shift);
    q += rq.output_zero_point;
    if (q < rq.activation_min) q = rq.activation_min;
    if (q > rq.activation_max) q = rq.activation_max;
    dst[c] = noodle_i8_clamp(q);
  }
  return C;
}

uint16_t noodle_i8_gmp(NoodleTensor *input, NoodleTensor *output) {
  if (!input || !output || !input->buffer.data) return 0;
  const uint32_t n = input->rank == NOODLE_TENSOR_2D
                   ? static_cast<uint32_t>(input->W) * input->W : input->W;
  int8_t *dst = noodle_i8_tensor_require_vector(output, input->C,
                                               input->scale, input->zero_point);
  if (!dst) return 0;
  for (uint16_t c = 0; c < input->C; ++c) {
    const int8_t *src = input->buffer.data + static_cast<uint32_t>(c) * n;
    int8_t v = src[0];
    for (uint32_t i = 1; i < n; ++i) if (src[i] > v) v = src[i];
    dst[c] = v;
  }
  return input->C;
}

uint16_t noodle_i8_flat(NoodleTensor *input, NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      input->rank != NOODLE_TENSOR_2D) return 0;
  const uint32_t plane = static_cast<uint32_t>(input->W) * input->W;
  const uint32_t total = plane * input->C;
  if (total > 65535u) return 0;
  int8_t *dst = noodle_i8_tensor_require_vector(output, static_cast<uint16_t>(total),
                                               input->scale, input->zero_point);
  if (!dst) return 0;
  for (uint16_t c = 0; c < input->C; ++c)
    for (uint32_t i = 0; i < plane; ++i)
      dst[i * input->C + c] = input->buffer.data[static_cast<uint32_t>(c) * plane + i];
  return static_cast<uint16_t>(total);
}

uint16_t noodle_i8_reshape_hwc_to_chw(NoodleTensor *input,
                                    uint16_t W, uint16_t C,
                                    NoodleTensor *output) {
  if (!input || !output || input == output || !input->buffer.data ||
      noodle_tensor_size(input) != static_cast<size_t>(W) * W * C) return 0;
  int8_t *dst = noodle_i8_tensor_require_2d(output, C, W,
                                          input->scale, input->zero_point);
  if (!dst) return 0;
  const uint32_t plane = static_cast<uint32_t>(W) * W;
  for (uint32_t pix = 0; pix < plane; ++pix)
    for (uint16_t c = 0; c < C; ++c)
      dst[static_cast<uint32_t>(c) * plane + pix] = input->buffer.data[pix * C + c];
  return static_cast<uint16_t>(plane * C);
}

uint16_t noodle_i8_concat(NoodleTensor *A, NoodleTensor *B,
                        NoodleTensor *output) {
  if (!A || !B || !output || !A->buffer.data || !B->buffer.data ||
      A->rank != B->rank || A->W != B->W || !q_same_params(A, B)) return 0;
  const uint16_t C = static_cast<uint16_t>(A->C + B->C);
  const size_t nA = noodle_tensor_size(A);
  const size_t nB = noodle_tensor_size(B);
  int8_t *dst = A->rank == NOODLE_TENSOR_2D
      ? noodle_i8_tensor_require_2d(output, C, A->W, A->scale, A->zero_point)
      : noodle_i8_tensor_require_1d(output, C, A->W, A->scale, A->zero_point);
  if (!dst) return 0;
  memcpy(dst, A->buffer.data, nA);
  memcpy(dst + nA, B->buffer.data, nB);
  return C;
}

uint16_t noodle_i8_relu(NoodleTensor *t) {
  if (!t || !t->buffer.data) return 0;
  const size_t n = noodle_tensor_size(t);
  const int8_t qzero = noodle_i8_clamp(t->zero_point);
  for (size_t i = 0; i < n; ++i) if (t->buffer.data[i] < qzero) t->buffer.data[i] = qzero;
  return n > 65535u ? 65535u : static_cast<uint16_t>(n);
}

void noodle_i8_find_max(const NoodleTensor *input, uint16_t *index,
                      int8_t *value) {
  if (index) *index = 0;
  if (value) *value = -128;
  if (!input || !input->buffer.data) return;
  const size_t n = noodle_tensor_size(input);
  if (!n) return;
  size_t best = 0;
  for (size_t i = 1; i < n; ++i) if (input->buffer.data[i] > input->buffer.data[best]) best = i;
  if (index) *index = best > 65535u ? 65535u : static_cast<uint16_t>(best);
  if (value) *value = input->buffer.data[best];
}

uint16_t noodle_i8_lut(NoodleTensor *input, const int8_t lut[256],
                     float output_scale, int32_t output_zp,
                     NoodleTensor *output) {
  if (!input || !output || !input->buffer.data || !lut) return 0;
  int8_t *dst = input->rank == NOODLE_TENSOR_2D
      ? noodle_i8_tensor_require_2d(output, input->C, input->W, output_scale, output_zp)
      : noodle_i8_tensor_require_1d(output, input->C, input->W, output_scale, output_zp);
  if (!dst) return 0;
  const size_t n = noodle_tensor_size(input);
  for (size_t i = 0; i < n; ++i) dst[i] = lut[static_cast<uint8_t>(input->buffer.data[i])];
  return n > 65535u ? 65535u : static_cast<uint16_t>(n);
}

uint16_t noodle_i8_softmax(NoodleTensor *input,
                         const NoodleI8SoftmaxLUT &lut,
                         NoodleTensor *output) {
  if (!input || !output || !input->buffer.data || !lut.exp_q15 ||
      input->rank != NOODLE_TENSOR_1D || input->W != 1 || input->C == 0 ||
      !q_scale_equal(input->scale, lut.input_scale)) return 0;
  int8_t *dst = noodle_i8_tensor_require_vector(output, input->C, 1.0f / 256.0f, -128);
  if (!dst) return 0;
  int8_t maxv = input->buffer.data[0];
  for (uint16_t i = 1; i < input->C; ++i) if (input->buffer.data[i] > maxv) maxv = input->buffer.data[i];
  uint64_t sum = 0;
  for (uint16_t i = 0; i < input->C; ++i) {
    const uint8_t d = static_cast<uint8_t>(static_cast<int32_t>(maxv) - input->buffer.data[i]);
    sum += lut.exp_q15[d];
  }
  if (!sum) return 0;
  for (uint16_t i = 0; i < input->C; ++i) {
    const uint8_t d = static_cast<uint8_t>(static_cast<int32_t>(maxv) - input->buffer.data[i]);
    const uint64_t num = static_cast<uint64_t>(lut.exp_q15[d]) * 256u + sum / 2u;
    int32_t p = static_cast<int32_t>(num / sum);
    if (p > 255) p = 255;
    dst[i] = static_cast<int8_t>(p - 128);
  }
  return input->C;
}

void noodle_build_softmax_lut(float input_scale, uint16_t lut[256]) {
  if (!lut) return;
  for (int d = 0; d < 256; ++d) {
    const double e = exp(-static_cast<double>(d) * input_scale);
    long q = lround(e * 32767.0);
    if (q < 0) q = 0;
    if (q > 32767) q = 32767;
    lut[d] = static_cast<uint16_t>(q);
  }
}

void noodle_build_sigmoid_lut(float input_scale, int32_t input_zp,
                               int8_t lut[256]) {
  if (!lut) return;
  for (int q = -128; q <= 127; ++q) {
    const double x = (q - input_zp) * static_cast<double>(input_scale);
    const double y = 1.0 / (1.0 + exp(-x));
    int32_t out = static_cast<int32_t>(lround(y * 256.0)) - 128;
    lut[static_cast<uint8_t>(static_cast<int8_t>(q))] = noodle_i8_clamp(out);
  }
}

void noodle_build_tanh_lut(float input_scale, int32_t input_zp,
                            int8_t lut[256]) {
  if (!lut) return;
  for (int q = -128; q <= 127; ++q) {
    const double x = (q - input_zp) * static_cast<double>(input_scale);
    int32_t out = static_cast<int32_t>(lround(tanh(x) * 128.0));
    lut[static_cast<uint8_t>(static_cast<int8_t>(q))] = noodle_i8_clamp(out);
  }
}

#endif  // NOODLE_USE_INT8
