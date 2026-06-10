/**
 * @file noodle_math.cpp
 * @brief Reusable math and numeric helper primitives.
 * @ingroup noodle_api
 */
#include "noodle_internal.h"


float noodle_dot_float_block(const float *x, const float *w, uint16_t n) {
  float s0 = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;
  float s3 = 0.0f;

  uint16_t i = 0;
  for (; (uint16_t)(i + 3) < n; i = (uint16_t)(i + 4)) {
    s0 += x[i + 0] * w[i + 0];
    s1 += x[i + 1] * w[i + 1];
    s2 += x[i + 2] * w[i + 2];
    s3 += x[i + 3] * w[i + 3];
  }

  float s = (s0 + s1) + (s2 + s3);
  for (; i < n; i++) {
    s += x[i] * w[i];
  }
  return s;
}


void noodle_find_max(float *input,
                     uint16_t n,
                     float &max_val,
                     uint16_t &max_idx) {
  max_val = input[0];
  max_idx = 0;
  for (uint16_t i = 1; i < n; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
      max_idx = i;
    }
  }
}

void noodle_unpack_bn_params(const float *bn_params,
                             uint16_t N,
                             const float **gamma,
                             const float **beta,
                             const float **mean,
                             const float **var) {
  *gamma = bn_params;
  *beta  = bn_params + N;
  *mean  = bn_params + 2 * N;
  *var   = bn_params + 3 * N;
}

uint16_t noodle_bn1d(float *x,
                     uint16_t N,
                     const float *gamma,
                     const float *beta,
                     const float *mean,
                     const float *var,
                     float eps) {
  for (uint16_t i = 0; i < N; ++i) {
    const float inv_std = 1.0f / sqrtf(var[i] + eps);
    const float s = gamma[i] * inv_std;
    const float t = beta[i] - s * mean[i];
    x[i] = s * x[i] + t;
  }
  return N;
}

uint16_t noodle_bn1d(float *x,
                     uint16_t N,
                     const float *bn_params,
                     float eps) {
  const float *gamma, *beta, *mean, *var;
  noodle_unpack_bn_params(bn_params, N, &gamma, &beta, &mean, &var);
  return noodle_bn1d(x, N, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn1d_relu(float *x,
                          uint16_t N,
                          const float *gamma,
                          const float *beta,
                          const float *mean,
                          const float *var,
                          float eps) {
  for (uint16_t i = 0; i < N; ++i) {
    const float inv_std = 1.0f / sqrtf(var[i] + eps);
    const float s = gamma[i] * inv_std;
    const float t = beta[i] - s * mean[i];
    const float y = s * x[i] + t;
    x[i] = (y > 0.0f) ? y : 0.0f;
  }
  return N;
}

uint16_t noodle_bn1d_relu(float *x,
                          uint16_t N,
                          const float *bn_params,
                          float eps) {
  const float *gamma, *beta, *mean, *var;
  noodle_unpack_bn_params(bn_params, N, &gamma, &beta, &mean, &var);
  return noodle_bn1d_relu(x, N, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn2d(float *x,
                     uint16_t C,
                     uint16_t W,
                     const float *gamma,
                     const float *beta,
                     const float *mean,
                     const float *var,
                     float eps) {
  const uint16_t plane = W * W;
  for (uint16_t c = 0; c < C; ++c) {
    const float inv_std = 1.0f / sqrtf(var[c] + eps);
    const float s = gamma[c] * inv_std;
    const float t = beta[c] - s * mean[c];

    float *p = x + (uint32_t)c * plane;
    for (uint16_t i = 0; i < plane; ++i) {
      p[i] = s * p[i] + t;
    }
  }
  return W;
}

uint16_t noodle_bn2d(float *x,
                     uint16_t C,
                     uint16_t W,
                     const float *bn_params,
                     float eps) {
  const float *gamma, *beta, *mean, *var;
  noodle_unpack_bn_params(bn_params, C, &gamma, &beta, &mean, &var);
  return noodle_bn2d(x, C, W, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn2d_relu(float *x,
                          uint16_t C,
                          uint16_t W,
                          const float *gamma,
                          const float *beta,
                          const float *mean,
                          const float *var,
                          float eps) {
  const uint16_t plane = W * W;
  for (uint16_t c = 0; c < C; ++c) {
    const float inv_std = 1.0f / sqrtf(var[c] + eps);
    const float s = gamma[c] * inv_std;
    const float t = beta[c] - s * mean[c];

    float *p = x + (uint32_t)c * plane;
    for (uint16_t i = 0; i < plane; ++i) {
      const float y = s * p[i] + t;
      p[i] = (y > 0.0f) ? y : 0.0f;
    }
  }
  return W;
}

uint16_t noodle_bn2d_relu(float *x,
                          uint16_t C,
                          uint16_t W,
                          const float *bn_params,
                          float eps) {
  const float *gamma, *beta, *mean, *var;
  noodle_unpack_bn_params(bn_params, C, &gamma, &beta, &mean, &var);
  return noodle_bn2d_relu(x, C, W, gamma, beta, mean, var, eps);
}

// Backward-compatible aliases. Existing code that calls noodle_bn(..., C, W, ...)
// keeps the old Conv2D/channel-first behavior. New code should prefer the
// explicit noodle_bn2d() or noodle_bn1d() names.
uint16_t noodle_bn(float *x,
                   uint16_t C,
                   uint16_t W,
                   const float *gamma,
                   const float *beta,
                   const float *mean,
                   const float *var,
                   float eps) {
  return noodle_bn2d(x, C, W, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn(float *x,
                   uint16_t C,
                   uint16_t W,
                   const float *bn_params,
                   float eps) {
  return noodle_bn2d(x, C, W, bn_params, eps);
}

uint16_t noodle_bn_relu(float *x,
                        uint16_t C,
                        uint16_t W,
                        const float *gamma,
                        const float *beta,
                        const float *mean,
                        const float *var,
                        float eps) {
  return noodle_bn2d_relu(x, C, W, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn_relu(float *x,
                        uint16_t C,
                        uint16_t W,
                        const float *bn_params,
                        float eps) {
  return noodle_bn2d_relu(x, C, W, bn_params, eps);
}

uint16_t noodle_soft_max(float *input_output,
                         uint16_t n) {
  float max_val = input_output[0];
  for (uint16_t i = 1; i < n; i++) {
    if (input_output[i] > max_val)
      max_val = input_output[i];
  }

  float sum = 0.0;
  for (uint16_t i = 0; i < n; i++) {
    input_output[i] = expf(input_output[i] - max_val);
    sum += input_output[i];
  }

  for (uint16_t i = 0; i < n; i++) {
    input_output[i] /= sum;
  }
  return n;
}

uint16_t noodle_sigmoid(float *input_output, 
                        uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    const float x = input_output[i];
    if (x >= 0.0f) {
      const float z = expf(-x);
      input_output[i] = 1.0f / (1.0f + z);
    } else {
      const float z = expf(x);
      input_output[i] = z / (1.0f + z);
    }
  }
  return n;
}

float noodle_sigmoidf(float x)
{
  if (x >= 0.0f) {
    const float z = expf(-x);
    return 1.0f / (1.0f + z);
  } else {
    const float z = expf(x);
    return z / (1.0f + z);
  }
}

uint16_t noodle_logit(float *input_output, 
                      uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    float v = input_output[i];
    if (v >= 0.0f) {
      float z = expf(-v);
      input_output[i] = 1.0f / (1.0f + z);
    } else {
      float z = expf(v);
      input_output[i] = z / (1.0f + z);
    }
  }
  return n;
}

uint16_t noodle_relu(float *input_output,
                     uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    input_output[i] = input_output[i] > 0.0f ? input_output[i] : 0.0f;
  }
  return n;
}

// ===== NoodleBuffer convenience wrappers =====

void noodle_find_max(NoodleBuffer *input,
                     uint16_t n,
                     float &max_val,
                     uint16_t &max_idx) {
  if (!input || !input->data || n == 0) {
    max_val = 0.0f;
    max_idx = 0;
    return;
  }
  noodle_find_max(input->data, n, max_val, max_idx);
}

uint16_t noodle_bn1d(NoodleBuffer *x,
                     uint16_t N,
                     const float *gamma,
                     const float *beta,
                     const float *mean,
                     const float *var,
                     float eps) {
  if (!x || !x->data) return 0;
  return noodle_bn1d(x->data, N, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn1d(NoodleBuffer *x,
                     uint16_t N,
                     const float *bn_params,
                     float eps) {
  if (!x || !x->data) return 0;
  return noodle_bn1d(x->data, N, bn_params, eps);
}

uint16_t noodle_bn1d_relu(NoodleBuffer *x,
                          uint16_t N,
                          const float *gamma,
                          const float *beta,
                          const float *mean,
                          const float *var,
                          float eps) {
  if (!x || !x->data) return 0;
  return noodle_bn1d_relu(x->data, N, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn1d_relu(NoodleBuffer *x,
                          uint16_t N,
                          const float *bn_params,
                          float eps) {
  if (!x || !x->data) return 0;
  return noodle_bn1d_relu(x->data, N, bn_params, eps);
}

uint16_t noodle_bn2d(NoodleBuffer *x,
                     uint16_t C,
                     uint16_t W,
                     const float *gamma,
                     const float *beta,
                     const float *mean,
                     const float *var,
                     float eps) {
  if (!x || !x->data) return 0;
  return noodle_bn2d(x->data, C, W, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn2d(NoodleBuffer *x,
                     uint16_t C,
                     uint16_t W,
                     const float *bn_params,
                     float eps) {
  if (!x || !x->data) return 0;
  return noodle_bn2d(x->data, C, W, bn_params, eps);
}

uint16_t noodle_bn2d_relu(NoodleBuffer *x,
                          uint16_t C,
                          uint16_t W,
                          const float *gamma,
                          const float *beta,
                          const float *mean,
                          const float *var,
                          float eps) {
  if (!x || !x->data) return 0;
  return noodle_bn2d_relu(x->data, C, W, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn2d_relu(NoodleBuffer *x,
                          uint16_t C,
                          uint16_t W,
                          const float *bn_params,
                          float eps) {
  if (!x || !x->data) return 0;
  return noodle_bn2d_relu(x->data, C, W, bn_params, eps);
}

uint16_t noodle_bn(NoodleBuffer *x,
                   uint16_t C,
                   uint16_t W,
                   const float *gamma,
                   const float *beta,
                   const float *mean,
                   const float *var,
                   float eps) {
  return noodle_bn2d(x, C, W, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn(NoodleBuffer *x,
                   uint16_t C,
                   uint16_t W,
                   const float *bn_params,
                   float eps) {
  return noodle_bn2d(x, C, W, bn_params, eps);
}

uint16_t noodle_bn_relu(NoodleBuffer *x,
                        uint16_t C,
                        uint16_t W,
                        const float *gamma,
                        const float *beta,
                        const float *mean,
                        const float *var,
                        float eps) {
  return noodle_bn2d_relu(x, C, W, gamma, beta, mean, var, eps);
}

uint16_t noodle_bn_relu(NoodleBuffer *x,
                        uint16_t C,
                        uint16_t W,
                        const float *bn_params,
                        float eps) {
  return noodle_bn2d_relu(x, C, W, bn_params, eps);
}

uint16_t noodle_soft_max(NoodleBuffer *input_output,
                         uint16_t n) {
  if (!input_output || !input_output->data) return 0;
  return noodle_soft_max(input_output->data, n);
}

uint16_t noodle_sigmoid(NoodleBuffer *input_output,
                        uint16_t n) {
  if (!input_output || !input_output->data) return 0;
  return noodle_sigmoid(input_output->data, n);
}

uint16_t noodle_logit(NoodleBuffer *input_output,
                      uint16_t n) {
  if (!input_output || !input_output->data) return 0;
  return noodle_logit(input_output->data, n);
}

uint16_t noodle_relu(NoodleBuffer *input_output,
                     uint16_t n) {
  if (!input_output || !input_output->data) return 0;
  return noodle_relu(input_output->data, n);
}
