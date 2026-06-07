/**
 * @file noodle_conv.cpp
 * @brief Public convolution layer wrappers: Conv1D, Conv2D, and ConvTranspose2D.
 *
 * Low-level kernels, padding, pooling, shape helpers, and other implementation
 * details live in noodle_internal.cpp and are declared in noodle_internal.h.
 */
#include "noodle_internal.h"

// ===== Conv1D layer APIs =====

uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const Conv &conv,
                       const Pool &pool,
                       CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0.0f;
  const uint16_t total = n_inputs * n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);      // packed input CHW
  fo = noodle_fs_open_write(out_fn);    // packed output CHW

  uint16_t V = 0;
  float kernel[NOODLE_MAX_K];

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W);
    const float bias = noodle_read_float(fb);

    // rewind packed input for each output channel
    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      for (uint16_t i = 0; i < W; i++) {
        in_buffer[i] = noodle_read_float(fi);
      }
      for (uint16_t k = 0; k < conv.K; k++) {
        kernel[k] = noodle_read_float(fw);
      }

      V = noodle_do_conv1d(in_buffer, kernel, W, conv.K, out_buffer, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    // Bias + activation in-place
    for (uint16_t i = 0; i < V; i++) {
      float v = out_buffer[i] + bias;
      if ((conv.act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      out_buffer[i] = v;
    }

    // Pool and append to packed output file
    V = noodle_do_pooling1d(out_buffer, V, pool.M, pool.T, fo);
  }

  fw.close();
  fb.close();
  fi.close();
  fo.close();
  return V;
}

uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const Conv &conv,
                       CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  float progress = 0.0f;
  const uint16_t total = n_inputs * n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);      // packed input CHW
  fo = noodle_fs_open_write(out_fn);    // packed output CHW

  uint16_t V = 0;
  float kernel[NOODLE_MAX_K];

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, W);
    const float bias = noodle_read_float(fb);

    // rewind packed input for each output channel
    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      for (uint16_t i = 0; i < W; i++) {
        in_buffer[i] = noodle_read_float(fi);
      }
      for (uint16_t k = 0; k < conv.K; k++) {
        kernel[k] = noodle_read_float(fw);
      }

      V = noodle_do_conv1d(in_buffer, kernel, W, conv.K, out_buffer, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    for (uint16_t i = 0; i < V; i++) {
      float v = out_buffer[i] + bias;
      if ((conv.act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      out_buffer[i] = v;
    }

    // No pooling: append raw V samples for this output channel.
    for (uint16_t i = 0; i < V; i++) {
      noodle_write_float(fo, out_buffer[i]);
    }
  }

  fw.close();
  fb.close();
  fi.close();
  fo.close();
  return V;
}

uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb) {
  if (!temp_buff1 || !temp_buff2) return 0;

  float *in_buffer  = (float *)temp_buff1;  // must hold W floats
  float *out_buffer = (float *)temp_buff2;  // must hold W floats

  //const uint16_t Vmax = (uint16_t)((W - conv.K + 2 * conv.P) / conv.S + 1);
  uint16_t V = 0;

  float progress = 0.0f;
  const uint16_t total = n_inputs * n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  fi = noodle_fs_open_read(in_fn);
  fo = noodle_fs_open_write(out_fn);

  for (uint16_t O = 0; O < n_outputs; O++) {
    //oodle_reset_buffer(out_buffer, Vmax);
    noodle_reset_buffer(out_buffer, W);
    const float bias = (conv.bias != nullptr) ? conv.bias[O] : 0.0f;
    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      const float *kptr = conv.weight + (O * n_inputs + I) * conv.K; // Conv1D stride

      for (uint16_t i = 0; i < W; i++) {
        in_buffer[i] = noodle_read_float(fi);
      }

      V = noodle_do_conv1d(in_buffer, (float *)kptr, W, conv.K, out_buffer, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    // Bias + activation for valid region
    for (uint16_t i = 0; i < V; i++) {
      float v = out_buffer[i] + bias;
      if ((conv.act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      out_buffer[i] = v;
    }

    for (uint16_t i = 0; i < V; i++) {
      noodle_write_float(fo, out_buffer[i]);
    }
  }

  fi.close();
  fo.close();
  return V;
}

uint16_t noodle_conv1d(float *in,
                       uint16_t n_inputs,
                       float *out,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb) {
  float *in_buffer = nullptr;
  float *out_buffer = nullptr;

  float progress = 0.0f;
  const uint16_t total = n_inputs * n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  const uint16_t Vmax = (uint16_t)((W - conv.K + 2 * conv.P) / conv.S + 1);
  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    // IMPORTANT: output channel stride must be Vmax, not W
    out_buffer = out + (size_t)O * Vmax;

    noodle_reset_buffer(out_buffer, Vmax);

    const float bias = conv.bias ? conv.bias[O] : 0.0f;

    for (uint16_t I = 0; I < n_inputs; I++) {
      // input is compact CHW with stride W
      in_buffer = in + (size_t)I * W;

      const float *kernel = conv.weight + (size_t)(O * n_inputs + I) * conv.K;

      V = noodle_do_conv1d(in_buffer, (float *)kernel, W, conv.K, out_buffer, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    for (uint16_t i = 0; i < V; i++) {
      float v = out_buffer[i] + bias;
      if ((conv.act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      out_buffer[i] = v;
    }
  }

  return V;
}

uint16_t noodle_conv1d(float *in,
                       uint16_t n_inputs,
                       float *out,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       const Pool &pool,
                       CBFPtr progress_cb) {
  if (!in || !out) return 0;
  if (!conv.weight) return 0;

  // temp_buff2 must be large enough to hold one output channel before pooling!
  if (!temp_buff2) return 0;

  float *conv_buffer = (float *)temp_buff2;

  const uint16_t Vmax = (uint16_t)((W - conv.K + 2 * conv.P) / conv.S + 1);

  uint16_t pool_M = pool.M;
  uint16_t pool_T = pool.T;

  if (pool_M <= 1) {
    pool_M = 1;
    pool_T = 1;
  }

  if (pool_T == 0) {
    pool_T = pool_M;
  }

  const uint16_t Wo = (pool_M <= 1) ? Vmax : (uint16_t)((Vmax - pool_M) / pool_T + 1);

  float progress = 0.0f;
  const uint16_t total = n_inputs * n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  uint16_t V = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    // Accumulate one full convolution output channel here.
    noodle_reset_buffer(conv_buffer, Vmax);

    const float bias = conv.bias ? conv.bias[O] : 0.0f;

    for (uint16_t I = 0; I < n_inputs; I++) {
      // Input is compact CHW.
      float *in_buffer = in + (size_t)I * W;

      // Weight layout is [O][I][K].
      const float *kernel = conv.weight + (size_t)(O * n_inputs + I) * conv.K;

      V = noodle_do_conv1d(in_buffer, (float *)kernel, W, conv.K, conv_buffer, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    // Bias + activation before pooling, matching Keras:
    // Conv1D(activation='relu') -> MaxPooling1D
    for (uint16_t i = 0; i < V; i++) {
      float v = conv_buffer[i] + bias;

      if ((conv.act == ACT_RELU) && (v < 0.0f)) {
        v = 0.0f;
      }

      conv_buffer[i] = v;
    }

    // Output is compact CHW with stride Wo, not W.
    float *out_ch = out + (size_t)O * Wo;

    if (pool_M <= 1) {
      for (uint16_t i = 0; i < V; i++) {
        out_ch[i] = conv_buffer[i];
      }
    } else {
      noodle_do_pooling1d(conv_buffer, V, pool_M, pool_T, out_ch);
    }
  }

  return Wo;
}

uint16_t noodle_conv1d(float *in,
                       uint16_t n_inputs,
                       const char *out_fn,
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb) {
  if (!temp_buff2) return 0; // need an accumulation buffer

  float *out_buffer = (float *)temp_buff2;

  float progress = 0.0f;
  const uint16_t total = n_inputs * n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  // Output length per channel
  //const uint16_t Vmax = (uint16_t)((W - conv.K + 2 * conv.P) / conv.S + 1);
  uint16_t V = 0;

  fo = noodle_fs_open_write(out_fn);  // packed output CHW (append per output channel)

  for (uint16_t O = 0; O < n_outputs; O++) {
    // reset accumulation buffer for this output channel
    //noodle_reset_buffer(out_buffer, Vmax);
    noodle_reset_buffer(out_buffer, W);

    const float bias = (conv.bias != nullptr) ? conv.bias[O] : 0.0f;

    for (uint16_t I = 0; I < n_inputs; I++) {
      const float *in_buffer = in + I * W;
      const float *kernel    = conv.weight + (O * n_inputs + I) * conv.K;

      // Accumulate into out_buffer
      V = noodle_do_conv1d((float *)in_buffer, (float *)kernel, W, conv.K, out_buffer, conv.P, conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    // Bias + activation in-place
    for (uint16_t i = 0; i < V; i++) {
      float v = out_buffer[i] + bias;
      if ((conv.act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      out_buffer[i] = v;
    }

    // Append raw V samples for this output channel
    for (uint16_t i = 0; i < V; i++) {
      noodle_write_float(fo, out_buffer[i]);
    }
  }

  fo.close();
  return V;
}

uint16_t noodle_conv1d(const char *in_fn,
                       uint16_t n_inputs,
                       float *out,              // packed output: [O][Vmax]
                       uint16_t n_outputs,
                       uint16_t W,
                       const ConvMem &conv,
                       CBFPtr progress_cb) {
  float *in_buffer = (float *)temp_buff1;

  float progress = 0.0f;
  const uint16_t total = n_inputs * n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  // Output length per channel
  const uint16_t Vmax = (uint16_t)((W - conv.K + 2 * conv.P) / conv.S + 1);
  uint16_t V = 0;

  fi = noodle_fs_open_read(in_fn);   // packed input CHW

  for (uint16_t O = 0; O < n_outputs; O++) {
    float *out_buffer = out + (uint32_t)O * Vmax; // slicing
    //noodle_reset_buffer(out_buffer, Vmax);
    noodle_reset_buffer(out_buffer, W);

    const float bias = (conv.bias != nullptr) ? conv.bias[O] : 0.0f;

    // Rewind packed input for each output channel
    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      // Read one input channel (W samples) into RAM buffer
      for (uint16_t i = 0; i < W; i++) {
        in_buffer[i] = noodle_read_float(fi);
      }

      const float *kernel = conv.weight + (O * n_inputs + I) * conv.K;

      // Accumulate into output channel buffer
      V = noodle_do_conv1d(in_buffer,
                           (float *)kernel,   // noodle_do_conv1d expects float*
                           W,
                           conv.K,
                           out_buffer,
                           conv.P,
                           conv.S);

      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    // Bias + activation in-place
    for (uint16_t i = 0; i < V; i++) {
      float v = out_buffer[i] + bias;
      if ((conv.act == ACT_RELU) && (v < 0.0f)) v = 0.0f;
      out_buffer[i] = v;
    }
  }

  fi.close();
  return V;
}


// ===== Conv2D layer APIs =====

uint16_t noodle_conv_byte(const char *in_fn,
                          uint16_t n_inputs,
                          uint16_t n_outputs,
                          const char *out_fn,
                          uint16_t W,
                          const Conv &conv,
                          const Pool &pool,
                          CBFPtr progress_cb) {
  byte  *in_buffer  = (byte *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  if (!in_buffer || !out_buffer) return 0;

  float progress = 0.0f;
  const uint16_t total = n_inputs * n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);      
  fo = noodle_fs_open_write(out_fn);      

  if (!fb || !fw || !fi || !fo) {
    if (fb) fb.close();
    if (fw) fw.close();
    if (fi) fi.close();
    if (fo) fo.close();
    return 0;
  }

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) {
    fw.close(); fb.close(); fi.close(); fo.close();
    return 0;
  }

  uint16_t Vout = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K]; 

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    const float bias = noodle_read_float(fb);
    noodle_rewind_file(fi); // rewind input file for each output channel
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_grid_from_file(fi, in_buffer, W);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) progress_cb(progress);
      progress += progress_step;
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, fo);
  }

  fw.close(); 
  fb.close(); 
  fi.close(); 
  fo.close();
  return Vout;
}

uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  if (!in_buffer || !out_buffer) return 0;

  float progress = 0.0f;
  float progress_step = 0.0f;
  if (progress_cb) {
    const uint16_t total = n_inputs * n_outputs;
    progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;
  }

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);
  fo = noodle_fs_open_write(out_fn);

  if (!fb || !fw || !fi || !fo) {
    if (fb) fb.close();
    if (fw) fw.close();
    if (fi) fi.close();
    if (fo) fo.close();
    return 0;
  }

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) {
    fw.close(); fb.close(); fi.close(); fo.close();
    return 0;
  }

  uint16_t Vout = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];
  
  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    const float bias = noodle_read_float(fb);
    noodle_rewind_file(fi); // rewind input file for each output channel
    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_grid_from_file(fi, in_buffer, W);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb){ 
        progress_cb(progress);
        progress += progress_step;
      }
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, fo);
  }
  fw.close();
  fb.close();
  fi.close();
  fo.close();
  return Vout;
}

uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const ConvMem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  if (!in_fn || !out_fn || !in_buffer || !out_buffer || !conv.weight) {
    return 0;
  }

  float progress = 0.0f;
  float progress_step = 0.0f;
  if (progress_cb) {
    const uint16_t total = n_inputs * n_outputs;
    progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;
  }

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

  for (uint16_t O = 0; O < n_outputs; O++) {
    // temp_buff2 holds one accumulated pre-pooling output plane.
    noodle_reset_buffer(out_buffer, (uint32_t)Vconv * Vconv);

    const float bias = (conv.bias != nullptr) ? conv.bias[O] : 0.0f;

    // Re-read all input channels for each output channel.
    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      // Read one input channel plane from file.
      noodle_grid_from_file(fi, in_buffer, W);

      // ConvMem weight layout:
      // [O][I][K][K]
      const float *kernel =
          conv.weight + (uint32_t)(O * n_inputs + I) * conv.K * conv.K;

      noodle_do_conv(in_buffer,
                     kernel,
                     conv.K,
                     W,
                     out_buffer,
                     conv.P,
                     conv.S);

      if (progress_cb) {
        progress_cb(progress);
        progress += progress_step;
      }
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);

    // Pool directly into output file.
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, fo);
  }

  fi.close();
  fo.close();

  return Vout;
}

uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  if (!in_buffer || !out_buffer || !output) return 0;

  float progress = 0.0f;
  float progress_step = 0.0f;
  if (progress_cb) {
    const uint16_t total = n_inputs * n_outputs;
    progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;
  }

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fi = noodle_fs_open_read(in_fn);  // packed input

  if (!fb || !fw || !fi) {
    if (fb) fb.close();
    if (fw) fw.close();
    if (fi) fi.close();
    return 0;
  }

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) {
    fw.close();
    fb.close();
    fi.close();
    return 0;
  }

  const uint16_t Wo = (uint16_t)((Vconv - pool.M) / pool.T + 1);
  uint16_t Vout = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  for (uint16_t O = 0; O < n_outputs; O++) {
    // temp_buff2 holds one pre-pooling output plane.
    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    const float bias = noodle_read_float(fb);

    // rewind packed input for each output filter
    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_grid_from_file(fi, in_buffer, W);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) {
        progress_cb(progress);
        progress += progress_step;
      }
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    float *out_plane = noodle_slice(output, Wo, O);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, out_plane);
  }

  fw.close(); 
  fb.close(); 
  fi.close();
  return Vout;
}

uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *out_buffer = (float *)temp_buff2;
  if (!input || !out_buffer) return 0;

  float progress = 0.0f;
  float progress_step = 0.0f;
  if (progress_cb){
    const uint16_t total = n_inputs * n_outputs;
    progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;
  }

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
  fo = noodle_fs_open_write(out_fn);   // packed output

  if (!fb || !fw || !fo) {
    if (fb) fb.close();
    if (fw) fw.close();
    if (fo) fo.close();
    return 0;
  }

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) {
    fw.close(); fb.close(); fo.close();
    return 0;
  }

  uint16_t Vout = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    const float bias = noodle_read_float(fb);

    for (uint16_t I = 0; I < n_inputs; I++) {
      float *in_buffer = noodle_slice(input, W, I);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);

      if (progress_cb) {
        progress_cb(progress);
        progress += progress_step;
      }
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, fo);
  }

  fw.close(); 
  fb.close(); 
  fo.close();
  return Vout;
}

uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const ConvMem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb)
{
  float *out_buffer = (float *)temp_buff2;
  if (!input || !out_buffer || !conv.weight) return 0;

  float progress = 0.0f;
  float progress_step = 0.0f;
  if (progress_cb){
    const uint16_t total = n_inputs * n_outputs;
    progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;
  }

  fo = noodle_fs_open_write(out_fn);
  if (!fo) return 0;

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) {
    fo.close();
    return 0;
  }

  uint16_t Vout = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    const float bias = conv.bias ? conv.bias[O] : 0.0f;

    // Accumulate over input channels
    for (uint16_t I = 0; I < n_inputs; I++) {
      const float *kernel = conv.weight + (uint32_t)(O * n_inputs + I) * conv.K * conv.K;
      float *in_plane = noodle_slice(input, W, I);  // expects CHW in memory
      noodle_do_conv(in_plane, kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) {
        progress_cb(progress);
        progress += progress_step;
      }
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, fo);
  }

  fo.close();
  return Vout;  // spatial size after pooling
}

uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer = nullptr;
  float *out_buffer = (float *)temp_buff2;

  if (!input || !output || !out_buffer) return 0;

  float progress = 0.0f;
  float progress_step = 0.0f;
  if (progress_cb) {
    const uint16_t total = n_inputs * n_outputs;
    progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;
  }

  fb = noodle_fs_open_read(conv.bias_fn);
  fw = noodle_fs_open_read(conv.weight_fn);
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
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  for (uint16_t O = 0; O < n_outputs; O++) {
    // temp_buff2 holds one pre-pooling output plane.
    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    const float bias = noodle_read_float(fb);

    for (uint16_t I = 0; I < n_inputs; I++) {
      in_buffer = noodle_slice(input, W, I);
      noodle_grid_from_file(fw, (float *)kernel, conv.K);
      noodle_do_conv(in_buffer, (float *)kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) {
        progress_cb(progress);
        progress += progress_step;
      }
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    float *out_plane = noodle_slice(output, Wo, O);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, out_plane);
  }

  fw.close();
  fb.close();
  return Vout;
}

uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const ConvMem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer = nullptr;
  float *out_buffer = (float *)temp_buff2;

  if (!input || !output || !out_buffer || !conv.weight) return 0;

  float progress = 0.0f;
  float progress_step = 0.0f;
  if (progress_cb) {
    const uint16_t total = n_inputs * n_outputs;
    progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;
  }

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) return 0;

  const uint16_t Wo = (uint16_t)((Vconv - pool.M) / pool.T + 1);
  uint16_t Vout = 0;

  for (uint16_t O = 0; O < n_outputs; O++) {
    // temp_buff2 holds one pre-pooling output plane.
    noodle_reset_buffer(out_buffer, Vconv * Vconv);
    const float bias = (conv.bias != nullptr) ? conv.bias[O] : 0.0f;

    for (uint16_t I = 0; I < n_inputs; I++) {
      const float *kernel = conv.weight + (uint32_t)(O * n_inputs + I) * conv.K * conv.K;
      in_buffer = noodle_slice(input, W, I);
      noodle_do_conv(in_buffer, kernel, conv.K, W, out_buffer, conv.P, conv.S);
      if (progress_cb) {
        progress_cb(progress);
        progress += progress_step;
      }
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    float *out_plane = noodle_slice(output, Wo, O);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, out_plane);
  }

  return Vout;
}


// ===== ConvTranspose2D layer APIs =====

uint16_t noodle_conv_transpose_float(float *input,
                                     uint16_t n_inputs,
                                     uint16_t n_outputs,
                                     float *output,
                                     uint16_t W,
                                     const ConvMem &conv,
                                     CBFPtr progress_cb) {
  if (!input || !output || !conv.weight) return 0;

  uint16_t P0, P1;
  const uint16_t Vt = noodle_compute_Vt_and_P(
      conv.K,
      W,
      conv.P,
      conv.S,
      conv.OP,
      P0,
      P1
  );

  if (Vt == 0) return 0;

  float progress = 0.0f;
  const uint32_t total = (uint32_t)n_inputs * (uint32_t)n_outputs;
  const float progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;

  const uint32_t plane = (uint32_t)Vt * (uint32_t)Vt;

  for (uint16_t O = 0; O < n_outputs; O++) {
    float *out_plane = noodle_slice(output, Vt, O);

    for (uint32_t i = 0; i < plane; i++) {
      out_plane[i] = 0.0f;
    }

    for (uint16_t I = 0; I < n_inputs; I++) {
      float *in_plane = noodle_slice(input, W, I);

      const float *kernel =
          conv.weight + ((uint32_t)O * n_inputs + I) * conv.K * conv.K;

      noodle_do_conv_transpose(in_plane, kernel, conv.K, W, out_plane, conv.P, conv.S, conv.OP);

      if (progress_cb) {
        progress_cb(progress);
        progress += progress_step;
      }
    }

    const float bias = conv.bias ? conv.bias[O] : 0.0f;

    for (uint32_t i = 0; i < plane; i++) {
      float v = out_plane[i] + bias;

      if ((conv.act == ACT_RELU) && (v < 0.0f)) {
        v = 0.0f;
      }

      out_plane[i] = v;
    }
  }

  if (progress_cb) progress_cb(1.0f);
  return Vt;
}

// File -> file normal Conv2D, PROGMEM parameters.
// Input layout:  [I][W][W]
// Weight layout: [O][I][K][K]
// Output layout: [O][Vout][Vout]
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const ConvProgmem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *in_buffer  = (float *)temp_buff1;
  float *out_buffer = (float *)temp_buff2;

  if (!in_fn || !out_fn || !in_buffer || !out_buffer || !conv.weight) return 0;

  float progress = 0.0f;
  float progress_step = 0.0f;
  if (progress_cb) {
    const uint16_t total = (uint16_t)(n_inputs * n_outputs);
    progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;
  }

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

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, (uint16_t)(Vconv * Vconv));

    const float bias = conv.bias ? noodle_pgm_float(conv.bias, O) : 0.0f;

    noodle_rewind_file(fi);

    for (uint16_t I = 0; I < n_inputs; I++) {
      noodle_grid_from_file(fi, in_buffer, W);

      const uint32_t kbase =
          ((uint32_t)O * (uint32_t)n_inputs + (uint32_t)I) *
          (uint32_t)conv.K * (uint32_t)conv.K;

      noodle_copy_kernel_progmem(conv.weight, kbase, conv.K, (float *)kernel);

      noodle_do_conv(in_buffer, (float *)kernel, conv.K, W,
                     out_buffer,
                     conv.P,
                     conv.S);

      if (progress_cb) {
        progress_cb(progress);
        progress += progress_step;
      }
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, fo);
  }

  fi.close();
  fo.close();

  return Vout;
}

// RAM -> RAM normal Conv2D, PROGMEM parameters.
// Input layout:  [I][W][W]
// Weight layout: [O][I][K][K]
// Output layout: [O][Vout][Vout]
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const ConvProgmem &conv,
                           const Pool &pool,
                           CBFPtr progress_cb) {
  float *out_buffer = (float *)temp_buff2;

  if (!input || !output || !out_buffer || !conv.weight) return 0;

  float progress = 0.0f;
  float progress_step = 0.0f;
  if (progress_cb) {
    const uint16_t total = (uint16_t)(n_inputs * n_outputs);
    progress_step = (total > 1) ? (1.0f / (float)(total - 1)) : 1.0f;
  }

  const uint16_t Vconv = noodle_compute_V(conv.K, W, conv.P, conv.S);
  if (Vconv == 0) return 0;

  const uint16_t Wo = (uint16_t)((Vconv - pool.M) / pool.T + 1);
  uint16_t Vout = 0;
  float kernel[NOODLE_MAX_K][NOODLE_MAX_K];

  for (uint16_t O = 0; O < n_outputs; O++) {
    noodle_reset_buffer(out_buffer, (uint16_t)(Vconv * Vconv));

    const float bias = conv.bias ? noodle_pgm_float(conv.bias, O) : 0.0f;

    for (uint16_t I = 0; I < n_inputs; I++) {
      float *in_plane = noodle_slice(input, W, I);

      const uint32_t kbase =
          ((uint32_t)O * (uint32_t)n_inputs + (uint32_t)I) *
          (uint32_t)conv.K * (uint32_t)conv.K;

      noodle_copy_kernel_progmem(conv.weight, kbase, conv.K, (float *)kernel);
      noodle_do_conv(in_plane, (float *)kernel,conv.K, W, out_buffer, conv.P, conv.S);

      if (progress_cb) {
        progress_cb(progress);
        progress += progress_step;
      }
    }

    noodle_do_bias_act(out_buffer, bias, Vconv, conv.act);

    float *out_plane = noodle_slice(output, Wo, O);
    Vout = noodle_do_pooling(out_buffer, Vconv, pool.M, pool.T, out_plane);
  }

  return Vout;
}
