/**
 * @file noodle_tensor.cpp
 * @brief Lightweight tensor metadata helpers and tensor-level wrapper API.
 * @ingroup noodle_api
 *
 * The tensor layer is intentionally thin: it validates NoodleTensor shape
 * metadata, delegates computation to the existing NoodleBuffer/raw kernels, and
 * updates output metadata only after the underlying operation succeeds.
 */

#include "noodle.h"

/**
 * @brief Grow tensor storage without changing logical shape metadata.
 *
 * @param t Tensor descriptor that owns the backing buffer.
 * @param required Required capacity in float elements.
 * @return Backing float pointer, or NULL on invalid input/allocation failure.
 */
static float *noodle_tensor_require_storage(NoodleTensor *t, size_t required) {
  if (!t || required == 0) return NULL;
  return noodle_buffer_require(&t->buffer, required);
}

void noodle_tensor_init(NoodleTensor *t) {
  if (!t) return;
  noodle_buffer_init(&t->buffer);
  t->C = 0;
  t->W = 0;
  t->rank = NOODLE_TENSOR_EMPTY;
}

void noodle_tensor_free(NoodleTensor *t) {
  if (!t) return;
  noodle_buffer_free(&t->buffer);
  t->C = 0;
  t->W = 0;
  t->rank = NOODLE_TENSOR_EMPTY;
}

float *noodle_tensor_require_1d(NoodleTensor *t, uint16_t C, uint16_t W) {
  if (!t || C == 0 || W == 0) return NULL;
  const size_t required = (size_t)C * (size_t)W;
  float *p = noodle_tensor_require_storage(t, required);
  if (!p) return NULL;
  t->C = C;
  t->W = W;
  t->rank = NOODLE_TENSOR_1D;
  return p;
}

float *noodle_tensor_require_2d(NoodleTensor *t, uint16_t C, uint16_t W) {
  if (!t || C == 0 || W == 0) return NULL;
  const size_t required = (size_t)C * (size_t)W * (size_t)W;
  float *p = noodle_tensor_require_storage(t, required);
  if (!p) return NULL;
  t->C = C;
  t->W = W;
  t->rank = NOODLE_TENSOR_2D;
  return p;
}

float *noodle_tensor_require_vector(NoodleTensor *t, uint16_t N) {
  return noodle_tensor_require_1d(t, N, 1);
}

size_t noodle_tensor_size(const NoodleTensor *t) {
  if (!t || t->rank == NOODLE_TENSOR_EMPTY) return 0;
  if (t->rank == NOODLE_TENSOR_1D) return (size_t)t->C * (size_t)t->W;
  if (t->rank == NOODLE_TENSOR_2D) return (size_t)t->C * (size_t)t->W * (size_t)t->W;
  return 0;
}

size_t noodle_tensor_capacity(const NoodleTensor *t) {
  if (!t) return 0;
  return noodle_buffer_capacity(&t->buffer);
}

size_t noodle_tensor_capacity_bytes(const NoodleTensor *t) {
  if (!t) return 0;
  return noodle_buffer_capacity_bytes(&t->buffer);
}

/**
 * @brief Check that a tensor has allocated packed `[C][W]` data.
 */
static bool noodle_tensor_valid_1d(const NoodleTensor *t) {
  return t && t->buffer.data && t->rank == NOODLE_TENSOR_1D && t->C > 0 && t->W > 0;
}

/**
 * @brief Check that a tensor has allocated packed `[C][W][W]` data.
 */
static bool noodle_tensor_valid_2d(const NoodleTensor *t) {
  return t && t->buffer.data && t->rank == NOODLE_TENSOR_2D && t->C > 0 && t->W > 0;
}

uint16_t noodle_conv2d(NoodleTensor *input,
                       NoodleTensor *output,
                       const Conv &conv,
                       const Pool &pool) {
  if (!noodle_tensor_valid_2d(input) || !output || conv.O == 0) return 0;

  const uint16_t Wout = noodle_conv_float(&input->buffer, input->C, conv.O, &output->buffer,
                                          input->W, conv, pool, NULL);
  if (Wout == 0) return 0;

  output->C = conv.O;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_2D;
  return Wout;
}

uint16_t noodle_conv2d(NoodleTensor *input,
                       NoodleTensor *output,
                       const ConvMem &conv,
                       const Pool &pool) {
  if (!noodle_tensor_valid_2d(input) || !output || conv.O == 0) return 0;

  const uint16_t Wout = noodle_conv_float(&input->buffer, input->C, conv.O, &output->buffer,
                                          input->W, conv, pool, NULL);
  if (Wout == 0) return 0;

  output->C = conv.O;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_2D;
  return Wout;
}

uint16_t noodle_conv2d(NoodleTensor *input,
                       NoodleTensor *output,
                       const ConvProgmem &conv,
                       const Pool &pool) {
  if (!noodle_tensor_valid_2d(input) || !output || conv.O == 0) return 0;

  const uint16_t Wout = noodle_conv_float(&input->buffer, input->C, conv.O, &output->buffer,
                                          input->W, conv, pool, NULL);
  if (Wout == 0) return 0;

  output->C = conv.O;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_2D;
  return Wout;
}

uint16_t noodle_conv_transpose2d(NoodleTensor *input,
                                 NoodleTensor *output,
                                 const ConvMem &conv) {
  if (!noodle_tensor_valid_2d(input) || !output || conv.O == 0) return 0;

  const uint16_t Wout = noodle_conv_transpose_float(&input->buffer, input->C, conv.O,
                                                    &output->buffer, input->W, conv, NULL);
  if (Wout == 0) return 0;

  output->C = conv.O;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_2D;
  return Wout;
}

uint16_t noodle_conv1d(NoodleTensor *input,
                       NoodleTensor *output,
                       const ConvMem &conv) {
  if (!noodle_tensor_valid_1d(input) || !output || conv.O == 0) return 0;

  const uint16_t Wout = noodle_conv1d(&input->buffer, input->C, &output->buffer, conv.O,
                                      input->W, conv, NULL);
  if (Wout == 0) return 0;

  output->C = conv.O;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_1D;
  return Wout;
}

uint16_t noodle_conv1d(NoodleTensor *input,
                       NoodleTensor *output,
                       const ConvMem &conv,
                       const Pool &pool) {
  if (!noodle_tensor_valid_1d(input) || !output || conv.O == 0) return 0;

  const uint16_t Wout = noodle_conv1d(&input->buffer, input->C, &output->buffer, conv.O,
                                      input->W, conv, pool, NULL);
  if (Wout == 0) return 0;

  output->C = conv.O;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_1D;
  return Wout;
}

uint16_t noodle_dwconv2d(NoodleTensor *input,
                         NoodleTensor *output,
                         const Conv &conv,
                         const Pool &pool) {
  if (!noodle_tensor_valid_2d(input) || !output) return 0;

  const uint16_t Wout = noodle_dwconv_float(&input->buffer, input->C, &output->buffer,
                                            input->W, conv, pool, NULL);
  if (Wout == 0) return 0;

  output->C = input->C;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_2D;
  return Wout;
}

uint16_t noodle_dwconv2d(NoodleTensor *input,
                         NoodleTensor *output,
                         const ConvMem &conv,
                         const Pool &pool) {
  if (!noodle_tensor_valid_2d(input) || !output) return 0;

  const uint16_t Wout = noodle_dwconv_float(&input->buffer, input->C, &output->buffer,
                                            input->W, conv, pool, NULL);
  if (Wout == 0) return 0;

  output->C = input->C;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_2D;
  return Wout;
}

uint16_t noodle_dwconv2d(NoodleTensor *input,
                         NoodleTensor *output,
                         const ConvProgmem &conv,
                         const Pool &pool) {
  if (!noodle_tensor_valid_2d(input) || !output) return 0;

  const uint16_t Wout = noodle_dwconv_float(&input->buffer, input->C, &output->buffer,
                                            input->W, conv, pool, NULL);
  if (Wout == 0) return 0;

  output->C = input->C;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_2D;
  return Wout;
}

uint16_t noodle_pool2d(NoodleTensor *input,
                       NoodleTensor *output,
                       uint16_t K,
                       uint16_t S) {
  if (!noodle_tensor_valid_2d(input) || !output) return 0;

  const uint16_t Wout = noodle_pool2d(&input->buffer, input->C, input->W, &output->buffer, K, S);
  if (Wout == 0) return 0;

  output->C = input->C;
  output->W = Wout;
  output->rank = NOODLE_TENSOR_2D;
  return Wout;
}

uint16_t noodle_gap(NoodleTensor *inout) {
  if (!noodle_tensor_valid_2d(inout)) return 0;
  const uint16_t n = noodle_gap(&inout->buffer, inout->C, inout->W);
  if (n == 0) return 0;
  inout->C = n;
  inout->W = 1;
  inout->rank = NOODLE_TENSOR_1D;
  return n;
}

uint16_t noodle_gmp(NoodleTensor *inout) {
  if (!noodle_tensor_valid_2d(inout)) return 0;
  const uint16_t n = noodle_gmp(&inout->buffer, inout->C, inout->W);
  if (n == 0) return 0;
  inout->C = n;
  inout->W = 1;
  inout->rank = NOODLE_TENSOR_1D;
  return n;
}

uint16_t noodle_flat(NoodleTensor *input,
                     NoodleTensor *output) {
  if (!noodle_tensor_valid_2d(input) || !output) return 0;

  const uint16_t n = noodle_flat(&input->buffer, &output->buffer, input->W, input->C);
  if (n == 0) return 0;

  output->C = n;
  output->W = 1;
  output->rank = NOODLE_TENSOR_1D;
  return n;
}

uint16_t noodle_concat(NoodleTensor *A,
                       NoodleTensor *B,
                       NoodleTensor *output) {
  if (!noodle_tensor_valid_2d(A) || !noodle_tensor_valid_2d(B) || !output) return 0;
  if (A->W != B->W) return 0;

  const uint16_t C = noodle_concat(&A->buffer, A->C, &B->buffer, B->C, &output->buffer, A->W);
  if (C == 0) return 0;

  output->C = C;
  output->W = A->W;
  output->rank = NOODLE_TENSOR_2D;
  return C;
}

uint16_t noodle_fcn(NoodleTensor *input,
                    NoodleTensor *output,
                    const FCNMem &fcn) {
  if (!noodle_tensor_valid_1d(input) || !output || fcn.O == 0) return 0;

  const uint16_t n = noodle_fcn(&input->buffer, (uint16_t)noodle_tensor_size(input),
                                fcn.O, &output->buffer, fcn, NULL);
  if (n == 0) return 0;

  output->C = n;
  output->W = 1;
  output->rank = NOODLE_TENSOR_1D;
  return n;
}

uint16_t noodle_fcn(NoodleTensor *input,
                    NoodleTensor *output,
                    const FCNFile &fcn) {
  if (!noodle_tensor_valid_1d(input) || !output || fcn.O == 0) return 0;

  const uint16_t n = noodle_fcn(&input->buffer, (uint16_t)noodle_tensor_size(input),
                                fcn.O, &output->buffer, fcn, NULL);
  if (n == 0) return 0;

  output->C = n;
  output->W = 1;
  output->rank = NOODLE_TENSOR_1D;
  return n;
}

uint16_t noodle_fcn(NoodleTensor *input,
                    NoodleTensor *output,
                    const FCNProgmem &fcn) {
  if (!noodle_tensor_valid_1d(input) || !output || fcn.O == 0) return 0;

  const uint16_t n = noodle_fcn(&input->buffer, (uint16_t)noodle_tensor_size(input),
                                fcn.O, &output->buffer, fcn, NULL);
  if (n == 0) return 0;

  output->C = n;
  output->W = 1;
  output->rank = NOODLE_TENSOR_1D;
  return n;
}

uint16_t noodle_soft_max(NoodleTensor *input_output) {
  if (!noodle_tensor_valid_1d(input_output)) return 0;
  return noodle_soft_max(&input_output->buffer, (uint16_t)noodle_tensor_size(input_output));
}

uint16_t noodle_sigmoid(NoodleTensor *input_output) {
  if (!input_output || !input_output->buffer.data) return 0;
  return noodle_sigmoid(&input_output->buffer, (uint16_t)noodle_tensor_size(input_output));
}

uint16_t noodle_relu(NoodleTensor *input_output) {
  if (!input_output || !input_output->buffer.data) return 0;
  return noodle_relu(&input_output->buffer, (uint16_t)noodle_tensor_size(input_output));
}
