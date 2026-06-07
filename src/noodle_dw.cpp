/**
 * @file noodle_dw.cpp
 * @brief Depthwise convolution operators.
 */
#include "noodle_internal.h"


// File CHW → File CHW with Conv
uint16_t noodle_dwconv_float(const char *in_fn,
                             uint16_t n_channels,
                             const char *out_fn,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb)
{
  float *in_buffer  = (float *)temp_buff1;  // holds one W*W plane
  float *out_buffer = (float *)temp_buff2;  // holds one Vconv*Vconv plane
  if (!in_buffer || !out_buffer) return 0;

  float progress = 0.0f;
  float progress_step = 1.0f / (float)(n_channels > 1 ? (n_channels - 1) : 1);

  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  fi = noodle_fs_open_read(in_fn);
  fb = noodle_fs_open_read(conv.bias_fn);    
  fw = noodle_fs_open_read(conv.weight_fn); 
  fo = noodle_fs_open_write(out_fn);

  if (!fi || !fb || !fw || !fo) {
    if (fi) fi.close();
    if (fb) fb.close();
    if (fw) fw.close();
    if (fo) fo.close();
    return 0;
  }

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) {
    fi.close(); fw.close(); fb.close(); fo.close();
    return 0;
  }

  uint16_t Vout = 0;

  for (uint16_t C = 0; C < n_channels; C++) {
    noodle_grid_from_file(fi, in_buffer, W);
    const float bias = noodle_read_float(fb);
    noodle_grid_from_file(fw, (float *)kernel, conv.K);

    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, fo);

    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fi.close();
  fw.close();
  fb.close();
  fo.close();

  return Vout;
}

// Memory CHW → Memory CHW with float input and Conv
uint16_t noodle_dwconv_float(float *input,
                             uint16_t n_channels,
                             float *output,
                             uint16_t W,
                             const Conv &conv,
                             const Pool &pool,
                             CBFPtr progress_cb)
{
  float *out_buffer = (float *)temp_buff2;
  if (!input || !output || !out_buffer) return 0;

  float progress = 0.0f;
  const float denom = (float)((n_channels > 1) ? (n_channels - 1) : 1);
  const float progress_step = (denom > 0.0f) ? (1.0f / denom) : 1.0f;

  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  NDL_File fb = noodle_fs_open_read(conv.bias_fn);
  NDL_File fw = noodle_fs_open_read(conv.weight_fn);
  if (!fb || !fw) {
    if (fb) fb.close();
    if (fw) fw.close();
    return 0;
  }

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) {
    fw.close();
    fb.close();
    return 0;
  }

  const uint16_t Wo = (uint16_t)((Vconv - pool.M) / pool.T + 1);
  uint16_t Vout = 0;

  for (uint16_t C = 0; C < n_channels; C++) {
    float *in_plane = noodle_slice(input, W, C);

    const float bias = noodle_read_float(fb);
    noodle_grid_from_file(fw, (float *)kernel, conv.K);

    // temp_buff2 holds one pre-pooling output plane.
    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    noodle_do_conv(in_plane, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    float *out_plane = noodle_slice(output, Wo, C);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, out_plane);

    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fw.close();
  fb.close();
  return Vout; 
}

uint16_t noodle_dwconv_float(float *input,
                             uint16_t n_channels,
                             float *output,
                             uint16_t W,
                             const ConvMem &conv,
                             const Pool &pool,
                             CBFPtr progress_cb)
{
  float *out_buffer = (float *)temp_buff2;
  if (!input || !output || !out_buffer || !conv.weight) return 0;

  float progress = 0.0f;
  const float denom = (float)((n_channels > 1) ? (n_channels - 1) : 1);
  const float progress_step = (denom > 0.0f) ? (1.0f / denom) : 1.0f;

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) return 0;

  const uint16_t Wo = (uint16_t)((Vconv - pool.M) / pool.T + 1);
  uint16_t Vout = 0;

  for (uint16_t C = 0; C < n_channels; C++) {
    float *in_plane = noodle_slice(input, W, C);

    const float bias = (conv.bias != nullptr) ? conv.bias[C] : 0.0f;
    const float *kernel = conv.weight + (uint32_t)C * conv.K * conv.K;

    // temp_buff2 holds one pre-pooling output plane.
    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    noodle_do_conv(in_plane, kernel, conv.K, W, out_buffer, conv.P, conv.S);

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    float *out_plane = noodle_slice(output, Wo, C);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, out_plane);

    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }
  return Vout;
}

// File -> file depthwise Conv2D, PROGMEM parameters.
// Input layout:  [C][W][W]
// Weight layout: [C][K][K]
// Output layout: [C][Vout][Vout]
uint16_t noodle_dwconv_float(const char *in_fn,
                             uint16_t C,
                             const char *out_fn,
                             uint16_t W,
                             const ConvProgmem &conv,
                             const Pool &pool,
                             CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  if (!in_fn || !out_fn || !in_buffer || !out_buffer || !conv.weight) return 0;

  float progress = 0.0f;
  const float progress_step = (C > 1) ? (1.0f / (float)(C - 1)) : 1.0f;

  fi = noodle_fs_open_read(in_fn);
  fo = noodle_fs_open_write(out_fn);

  if (!fi || !fo) {
    if (fi) fi.close();
    if (fo) fo.close();
    return 0;
  }

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) {
    fi.close();
    fo.close();
    return 0;
  }

  uint16_t Vout = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  for (uint16_t c = 0; c < C; c++) {
    noodle_grid_from_file(fi, in_buffer, W);
    noodle_reset_buffer(out_buffer, (uint16_t)(Vconv * Vconv));

    const uint32_t kbase =
        (uint32_t)c * (uint32_t)conv.K * (uint32_t)conv.K;

    noodle_copy_kernel_progmem(conv.weight, kbase, conv.K, (float *)kernel);
    noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);

    const float bias = conv.bias ? noodle_pgm_float(conv.bias, c) : 0.0f;
    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);

    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, fo);

    if (progress_cb) {
      progress_cb(progress);
      progress += progress_step;
    }
  }

  fi.close();
  fo.close();

  return Vout;
}

// RAM -> RAM depthwise Conv2D, PROGMEM parameters.
// Input layout:  [C][W][W]
// Weight layout: [C][K][K]
// Output layout: [C][Vout][Vout]
uint16_t noodle_dwconv_float(float *input,
                             uint16_t C,
                             float *output,
                             uint16_t W,
                             const ConvProgmem &conv,
                             const Pool &pool,
                             CBFPtr progress_cb) {
  float *out_buffer = (float *)temp_buff2;

  if (!input || !output || !out_buffer || !conv.weight) return 0;

  float progress = 0.0f;
  const float progress_step = (C > 1) ? (1.0f / (float)(C - 1)) : 1.0f;

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) return 0;

  const uint16_t Wo = (uint16_t)((Vconv - pool.M) / pool.T + 1);
  uint16_t Vout = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  for (uint16_t c = 0; c < C; c++) {
    float *in_plane  = noodle_slice(input, W, c);
    float *out_plane = noodle_slice(output, Wo, c);

    noodle_reset_buffer(out_buffer, (uint16_t)(Vconv * Vconv));

    const uint32_t kbase =
        (uint32_t)c * (uint32_t)conv.K * (uint32_t)conv.K;

    noodle_copy_kernel_progmem(conv.weight, kbase, conv.K, (float *)kernel);
    noodle_do_conv(in_plane, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);

    const float bias = conv.bias ? noodle_pgm_float(conv.bias, c) : 0.0f;
    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);

    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, out_plane);

    if (progress_cb) {
      progress_cb(progress);
      progress += progress_step;
    }
  }

  return Vout;
}