/**
 * @file noodle_fcn.cpp
 * @brief Fully connected layers.
 */
#include "noodle_internal.h"


uint16_t noodle_fcn(const int8_t *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0.0;
  float progress_step = 1.0f / (float)(n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fo = noodle_fs_open_write(out_fn);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      h += (float)input[j] * noodle_read_float(fw);
    if ((h < 0.0) && (fcn.act == ACT_RELU)) h = 0.0;
    noodle_write_float(fo, h);
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

// Memory HWC-flatten → File HWC-flatten
uint16_t noodle_fcn(const byte *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fo = noodle_fs_open_write(out_fn);

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      h += (float)input[j] * noodle_read_float(fw);
    if ((h < 0.0) && (fcn.act == ACT_RELU)) h = 0.0;
    noodle_write_float(fo, h);
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

// Memory HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const byte *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);

  for (uint16_t k = 0; k < n_outputs; k++) {
    output[k] = noodle_read_float(fb);
    for (uint16_t j = 0; j < n_inputs; j++)
      output[k] += (float)input[j] * noodle_read_float(fw);
    if ((fcn.act == ACT_RELU) && (output[k] < 0.f)) output[k] = 0.0f;
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fw.close();
  fb.close();

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);

  return n_outputs;
}

// Memory HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const float *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  if (!input || !output) return 0;

  float progress = 0.0f;
  const float progress_step = (n_outputs > 1)
                                ? (1.0f / (float)(n_outputs - 1))
                                : 1.0f;

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);

  if (!fw || !fb) {
    if (fw) fw.close();
    if (fb) fb.close();
    return 0;
  }

  float wbuf[NOODLE_FCN_BLOCK];

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);

    uint16_t j = 0;
    while (j < n_inputs) {
      const uint16_t remain = (uint16_t)(n_inputs - j);
      const uint16_t nb = (remain > (uint16_t)NOODLE_FCN_BLOCK)
                            ? (uint16_t)NOODLE_FCN_BLOCK
                            : remain;

      if (noodle_read_float_block(fw, wbuf, nb) != nb) {
        fw.close();
        fb.close();
        return 0;
      }

      h += noodle_dot_float_block(input + j, wbuf, nb);
      j = (uint16_t)(j + nb);
    }

    if ((fcn.act == ACT_RELU) && (h < 0.0f)) h = 0.0f;
    output[k] = h;

    if (progress_cb) {
      progress_cb(progress);
      progress += progress_step;
    }
  }

  fw.close();
  fb.close();

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);

  return n_outputs;
}

// Memory HWC-flatten → File (streamed outputs)
uint16_t noodle_fcn(const float *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb)
{
  if (!input) return 0;

  float progress = 0.0f;
  const float progress_step = (n_outputs > 1)
                                ? (1.0f / (float)(n_outputs - 1))
                                : 1.0f;

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fo = noodle_fs_open_write(out_fn);

  if (!fw || !fb || !fo) {
    if (fw) fw.close();
    if (fb) fb.close();
    if (fo) fo.close();
    return 0;
  }

  float wbuf[NOODLE_FCN_BLOCK];

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = noodle_read_float(fb);

    uint16_t j = 0;
    while (j < n_inputs) {
      const uint16_t remain = (uint16_t)(n_inputs - j);
      const uint16_t nb = (remain > (uint16_t)NOODLE_FCN_BLOCK)
                            ? (uint16_t)NOODLE_FCN_BLOCK
                            : remain;

      if (noodle_read_float_block(fw, wbuf, nb) != nb) {
        fw.close();
        fb.close();
        fo.close();
        return 0;
      }

      h += noodle_dot_float_block(input + j, wbuf, nb);
      j = (uint16_t)(j + nb);
    }

    if ((fcn.act == ACT_RELU) && (h < 0.0f)) h = 0.0f;

    noodle_write_float(fo, h);

    if (progress_cb) {
      progress_cb(progress);
      progress += progress_step;
    }
  }

  fw.close();
  fb.close();
  fo.close();
  return n_outputs;
}

// File HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fi = noodle_fs_open_read(in_fn);

  for (uint16_t j = 0; j < n_outputs; j++) {
    output[j] = noodle_read_float(fb);
    noodle_rewind_file(fi);
    for (uint16_t k = 0; k < n_inputs; k++)
      output[j] += noodle_read_float(fi) * noodle_read_float(fw);
    if ((output[j] < 0.0) && (fcn.act == ACT_RELU)) output[j] = 0.0;
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fi.close();
  fw.close();
  fb.close();

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);

  return n_outputs;
}

// Memory HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const int8_t *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);

  for (uint16_t j = 0; j < n_outputs; j++) {
    output[j] = noodle_read_float(fb);
    for (uint16_t k = 0; k < n_inputs; k++)
      output[j] += (float)input[k] * noodle_read_float(fw);
    if ((output[j] < 0.0) && (fcn.act == ACT_RELU)) output[j] = 0.0;
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fw.close();
  fb.close();

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);

  return n_outputs;
}

// File HWC-flatten → File HWC-flatten
uint16_t noodle_fcn(const char *in_fn,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    const char *out_fn,
                    const FCNFile &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_inputs * n_outputs - 1);

  fw = noodle_fs_open_read(fcn.weight_fn);
  fb = noodle_fs_open_read(fcn.bias_fn);
  fo = noodle_fs_open_write(out_fn);
  fi = noodle_fs_open_read(in_fn);

  for (uint16_t j = 0; j < n_outputs; j++) {
    float h = noodle_read_float(fb);
    noodle_rewind_file(fi);
    for (uint16_t k = 0; k < n_inputs; k++)
      h += noodle_read_float(fi) * noodle_read_float(fw);
    if ((h < 0.0) && (fcn.act == ACT_RELU)) h = 0.0;
    noodle_write_float(fo, h);
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  fi.close();
  fo.close();
  fw.close();
  fb.close();
  return n_outputs;
}

// Memory HWC-flatten → Memory HWC-flatten
uint16_t noodle_fcn(const float *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNMem &fcn,
                    CBFPtr progress_cb) {
  float progress = 0;
  float progress_step = 1.0f / (float)(n_outputs - 1);

  uint16_t l = 0;
  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = fcn.bias ? fcn.bias[k] : 0.0f;
    for (uint16_t j = 0; j < n_inputs; j++)
      h += input[j] * fcn.weight[l++];  // <-- was ++l
    if ((fcn.act == ACT_RELU) && (h < 0.f)) h = 0.f;
    output[k] = h;
    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  if (fcn.act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);
  
  return n_outputs;
}

// Memory HWC-flatten -> Memory HWC-flatten with AVR PROGMEM weights
#if defined(__AVR__)
uint16_t noodle_fcn(const float *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNProgmem &fcn,
                    CBFPtr progress_cb) {
  if (!input || !output || fcn.weight_far == 0) return 0;

  float progress = 0.0f;
  float progress_step = (n_outputs > 1) ? (1.0f / (float)(n_outputs - 1)) : 1.0f;

  uint32_t l = 0;

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = 0.0f;

    if (fcn.bias_far != 0) {
      h = pgm_read_float_far(fcn.bias_far + (uint32_t)k * sizeof(float));
    }

    for (uint16_t j = 0; j < n_inputs; j++) {
      const float w = pgm_read_float_far(
        fcn.weight_far + l * sizeof(float)
      );
      h += input[j] * w;
      l++;
    }

    if ((fcn.act == ACT_RELU) && (h < 0.0f)) h = 0.0f;

    output[k] = h;

    if (progress_cb) progress_cb(progress);
    progress += progress_step;
  }

  if (fcn.act == ACT_SOFTMAX) {
    noodle_soft_max(output, n_outputs);
  }

  return n_outputs;
}

#else
uint16_t noodle_fcn(const float *input,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    float *output,
                    const FCNProgmem &fcn,
                    CBFPtr progress_cb) {
  (void)input; (void)n_inputs; (void)n_outputs; (void)output; (void)fcn; (void)progress_cb;
  return 0;
}
#endif

uint16_t noodle_fcn_progmem(const float *input,
                            uint16_t n_inputs,
                            uint16_t n_outputs,
                            float *output,
                            const float *weight,
                            const float *bias,
                            Activation act,
                            CBFPtr progress_cb) {
  if (!input || !output || !weight) return 0;

  float progress = 0.0f;
  const float progress_step = (n_outputs > 1)
                                ? (1.0f / (float)(n_outputs - 1))
                                : 1.0f;

  for (uint16_t k = 0; k < n_outputs; k++) {
    float h = bias ? noodle_pgm_float(bias, k) : 0.0f;

    const uint32_t row = (uint32_t)k * (uint32_t)n_inputs;
    for (uint16_t j = 0; j < n_inputs; j++) {
      h += input[j] * noodle_pgm_float(weight, row + j);
    }

    if ((act == ACT_RELU) && (h < 0.0f)) h = 0.0f;
    output[k] = h;

    if (progress_cb) {
      progress_cb(progress);
      progress += progress_step;
    }
  }

  if (act == ACT_SOFTMAX) noodle_soft_max(output, n_outputs);
  return n_outputs;
}
