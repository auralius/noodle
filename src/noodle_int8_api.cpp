#include "noodle_config.h"
#if defined(NOODLE_USE_INT8)

#include "noodle_int8.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void noodle_setup_temp_buffers(void *b1, void *b2) {
  (void)b1;
  (void)b2;
  // The full-int8 kernels own correctly typed int8/int32 scratch storage.
}

void noodle_setup_temp_buffers(void *b2) {
  (void)b2;
}

void noodle_temp_buffers_free(void) {
  noodle_i8_temp_buffers_free();
}

float *noodle_create_buffer(uint16_t size) {
  return (float *)malloc(size);
}

void noodle_delete_buffer(float *buffer) {
  free(buffer);
}


static NoodleI8Pool make_pool(const Pool &pool) {
  NoodleI8Pool p;
  p.M = pool.M;
  p.T = pool.T;
#if NOODLE_POOL_MODE == NOODLE_POOL_MEAN
  p.mode = NOODLE_I8_POOL_MEAN;
#else
  p.mode = NOODLE_I8_POOL_MAX;
#endif
  return p;
}

static bool identity_pool(const Pool &p) {
  return p.M == 1 && p.T == 1;
}

template <typename Layer>
static Layer with_activation_bounds(const Layer &src) {
  Layer dst = src;
  if (dst.act == ACT_RELU) {
    const int32_t qzero = noodle_activation_min_relu(
        dst.output_scale, dst.output_zero_point);
    if (dst.activation_min < qzero) dst.activation_min = qzero;
  }
  return dst;
}

static uint16_t copy_tensor_result(NoodleTensor *src, NoodleTensor *dst) {
  if (!src || !dst || !src->buffer.data) return 0;
  const size_t n = noodle_tensor_size(src);
  NoodleData *p = src->rank == NOODLE_TENSOR_2D
      ? noodle_i8_tensor_require_2d(dst, src->C, src->W, src->scale, src->zero_point)
      : noodle_i8_tensor_require_1d(dst, src->C, src->W, src->scale, src->zero_point);
  if (!p) return 0;
  memmove(p, src->buffer.data, n * sizeof(NoodleData));
  return src->rank == NOODLE_TENSOR_2D ? src->W : src->C;
}

void noodle_tensor_set_quantization(NoodleTensor *t, float scale,
                                    int32_t zero_point) {
  if (!t || !(scale > 0.0f) || zero_point < -128 || zero_point > 127) return;
  t->scale = scale;
  t->zero_point = zero_point;
}

NoodleData *noodle_tensor_require_1d(NoodleTensor *t, uint16_t C, uint16_t W) {
  const float scale = t && t->scale > 0.0f ? t->scale : 1.0f;
  const int32_t zp = t ? t->zero_point : 0;
  return noodle_i8_tensor_require_1d(t, C, W, scale, zp);
}

NoodleData *noodle_tensor_require_2d(NoodleTensor *t, uint16_t C, uint16_t W) {
  const float scale = t && t->scale > 0.0f ? t->scale : 1.0f;
  const int32_t zp = t ? t->zero_point : 0;
  return noodle_i8_tensor_require_2d(t, C, W, scale, zp);
}

NoodleData *noodle_tensor_require_vector(NoodleTensor *t, uint16_t N) {
  return noodle_tensor_require_1d(t, N, 1);
}

template <typename Layer>
static uint16_t conv2d_with_pool(NoodleTensor *input, NoodleTensor *output,
                                 const Layer &layer, const Pool &pool) {
  Layer configured = with_activation_bounds(layer);
  if (identity_pool(pool)) return noodle_i8_conv2d(input, configured, output);

  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t v = noodle_i8_conv2d(input, configured, &tmp);
  if (!v) { noodle_tensor_free(&tmp); return 0; }
  const uint16_t out = noodle_i8_pool2d(&tmp, make_pool(pool), output);
  noodle_tensor_free(&tmp);
  return out;
}

uint16_t noodle_conv2d(NoodleTensor *input, NoodleTensor *output,
                       const Conv &conv, const Pool &pool) {
  return conv2d_with_pool(input, output, conv, pool);
}

uint16_t noodle_conv2d(NoodleTensor *input, NoodleTensor *output,
                       const ConvMem &conv, const Pool &pool) {
  return conv2d_with_pool(input, output, conv, pool);
}

uint16_t noodle_conv2d(NoodleTensor *input, NoodleTensor *output,
                       const ConvProgmem &conv, const Pool &pool) {
  ConvProgmem configured = with_activation_bounds(conv);
  if (identity_pool(pool)) return noodle_i8_conv2d(input, configured, output);
  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t v = noodle_i8_conv2d(input, configured, &tmp);
  if (!v) { noodle_tensor_free(&tmp); return 0; }
  const uint16_t out = noodle_i8_pool2d(&tmp, make_pool(pool), output);
  noodle_tensor_free(&tmp);
  return out;
}

uint16_t noodle_conv_transpose2d(NoodleTensor *input, NoodleTensor *output,
                                 const ConvMem &conv) {
  ConvMem configured = with_activation_bounds(conv);
  return noodle_i8_conv_transpose2d(input, configured, output);
}

uint16_t noodle_conv1d(NoodleTensor *input, NoodleTensor *output,
                       const ConvMem &conv) {
  ConvMem configured = with_activation_bounds(conv);
  return noodle_i8_conv1d(input, configured, output);
}

uint16_t noodle_conv1d(NoodleTensor *input, NoodleTensor *output,
                       const ConvMem &conv, const Pool &pool) {
  ConvMem configured = with_activation_bounds(conv);
  if (identity_pool(pool)) return noodle_i8_conv1d(input, configured, output);
  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t v = noodle_i8_conv1d(input, configured, &tmp);
  if (!v) { noodle_tensor_free(&tmp); return 0; }
  const uint16_t out = noodle_i8_pool1d(&tmp, make_pool(pool), output);
  noodle_tensor_free(&tmp);
  return out;
}

template <typename Layer>
static uint16_t dwconv_with_pool(NoodleTensor *input, NoodleTensor *output,
                                 const Layer &layer, const Pool &pool) {
  Layer configured = with_activation_bounds(layer);
  if (identity_pool(pool)) return noodle_i8_dwconv2d(input, configured, output);
  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t v = noodle_i8_dwconv2d(input, configured, &tmp);
  if (!v) { noodle_tensor_free(&tmp); return 0; }
  const uint16_t out = noodle_i8_pool2d(&tmp, make_pool(pool), output);
  noodle_tensor_free(&tmp);
  return out;
}

uint16_t noodle_dwconv2d(NoodleTensor *input, NoodleTensor *output,
                         const Conv &conv, const Pool &pool) {
  return dwconv_with_pool(input, output, conv, pool);
}

uint16_t noodle_dwconv2d(NoodleTensor *input, NoodleTensor *output,
                         const ConvMem &conv, const Pool &pool) {
  return dwconv_with_pool(input, output, conv, pool);
}

uint16_t noodle_dwconv2d(NoodleTensor *input, NoodleTensor *output,
                         const ConvProgmem &conv, const Pool &pool) {
  ConvProgmem configured = with_activation_bounds(conv);
  if (identity_pool(pool)) return noodle_i8_dwconv2d(input, configured, output);
  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t v = noodle_i8_dwconv2d(input, configured, &tmp);
  if (!v) { noodle_tensor_free(&tmp); return 0; }
  const uint16_t out = noodle_i8_pool2d(&tmp, make_pool(pool), output);
  noodle_tensor_free(&tmp);
  return out;
}

uint16_t noodle_pool2d(NoodleTensor *input, NoodleTensor *output,
                       uint16_t K, uint16_t S) {
  Pool p; p.M = K; p.T = S;
  return noodle_i8_pool2d(input, make_pool(p), output);
}

uint16_t noodle_gap(NoodleTensor *inout) {
  if (!inout || inout->rank != NOODLE_TENSOR_2D || !inout->buffer.data) return 0;
  const uint32_t n = (uint32_t)inout->W * inout->W;
  NoodleI8Requant rq;
  rq.output_scale = inout->scale;
  rq.output_zero_point = inout->zero_point;
  rq.activation_min = -128;
  rq.activation_max = 127;
  if (!noodle_quantize_multiplier(1.0 / (double)n, &rq.multiplier, &rq.shift)) return 0;
  return noodle_i8_gap(inout, rq, inout);
}

uint16_t noodle_gmp(NoodleTensor *inout) {
  return noodle_i8_gmp(inout, inout);
}

uint16_t noodle_flat(NoodleTensor *input, NoodleTensor *output) {
  return noodle_i8_flat(input, output);
}

uint16_t noodle_concat(NoodleTensor *A, NoodleTensor *B, NoodleTensor *output) {
  return noodle_i8_concat(A, B, output);
}

uint16_t noodle_fcn(NoodleTensor *input, NoodleTensor *output,
                    const FCNMem &fcn) {
  FCNMem configured = with_activation_bounds(fcn);
  return noodle_i8_fcn(input, configured, output);
}

uint16_t noodle_fcn(NoodleTensor *input, NoodleTensor *output,
                    const FCNFile &fcn) {
  FCNFile configured = with_activation_bounds(fcn);
  return noodle_i8_fcn(input, configured, output);
}

uint16_t noodle_fcn(NoodleTensor *input, NoodleTensor *output,
                    const FCNProgmem &fcn) {
  FCNProgmem configured = fcn;
  if ((Activation)configured.act == ACT_RELU) {
    const int32_t qzero = noodle_activation_min_relu(
        configured.output_scale, configured.output_zero_point);
    if (configured.activation_min < qzero) configured.activation_min = qzero;
  }
  return noodle_i8_fcn(input, configured, output);
}

uint16_t noodle_soft_max(NoodleTensor *input_output) {
  if (!input_output || input_output->rank != NOODLE_TENSOR_1D) return 0;
  uint16_t lut_data[256];
  noodle_build_softmax_lut(input_output->scale, lut_data);
  NoodleI8SoftmaxLUT lut;
  lut.exp_q15 = lut_data;
  lut.input_scale = input_output->scale;
  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t n = noodle_i8_softmax(input_output, lut, &tmp);
  if (n) copy_tensor_result(&tmp, input_output);
  noodle_tensor_free(&tmp);
  return n;
}

uint16_t noodle_sigmoid(NoodleTensor *input_output) {
  if (!input_output || !input_output->buffer.data) return 0;
  int8_t lut[256];
  noodle_build_sigmoid_lut(input_output->scale, input_output->zero_point, lut);
  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t n = noodle_i8_lut(input_output, lut, 1.0f / 256.0f, -128, &tmp);
  if (n) copy_tensor_result(&tmp, input_output);
  noodle_tensor_free(&tmp);
  return n;
}

uint16_t noodle_relu(NoodleTensor *input_output) {
  return noodle_i8_relu(input_output);
}

#endif  // NOODLE_USE_INT8
