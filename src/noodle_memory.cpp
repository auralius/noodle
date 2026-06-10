/**
 * @file noodle_memory.cpp
 * @brief Noodle memory access management helpers.
 * @ingroup noodle_api
 */
#include "noodle_internal.h"
#include <limits.h>

#if defined(ARDUINO_ARCH_ESP32)
#include <Arduino.h>
#endif

extern size_t temp_buff1_capacity;
extern size_t temp_buff2_capacity;

#ifndef NOODLE_TEMP_EXTERNAL_CAPACITY_UNKNOWN
#define NOODLE_TEMP_EXTERNAL_CAPACITY_UNKNOWN ((size_t)-1)
#endif

static float *noodle_alloc_temp_float(size_t required_floats) {
  if (required_floats == 0) return NULL;

  const size_t bytes = required_floats * sizeof(float);

  // Scratch buffers are accessed frequently, so prefer internal RAM first.
  float *p = (float *)malloc(bytes);

#if defined(ARDUINO_ARCH_ESP32)
  if (!p && psramFound()) {
    p = (float *)ps_malloc(bytes);
  }
#endif

  return p;
}

static float *noodle_temp_require_impl(void **ptr,
                                       size_t *capacity,
                                       size_t required_floats) {
  if (!ptr || !capacity || required_floats == 0) return NULL;

  if (*ptr && *capacity >= required_floats) {
    return (float *)(*ptr);
  }

  // If the user installed a legacy external temp buffer with unknown capacity,
  // we cannot safely resize or free it. Keep old behavior and fail only if
  // Noodle knows it is too small. With unknown capacity we assume the user knew.
  if (*ptr && *capacity == NOODLE_TEMP_EXTERNAL_CAPACITY_UNKNOWN) {
    return (float *)(*ptr);
  }

  float *new_ptr = noodle_alloc_temp_float(required_floats);
  if (!new_ptr) {
    return NULL;
  }

  if (*ptr) {
    free(*ptr);
  }

  *ptr = new_ptr;
  *capacity = required_floats;
  return new_ptr;
}

float *noodle_temp1_require(size_t required_floats) {
  return noodle_temp_require_impl(&temp_buff1, &temp_buff1_capacity, required_floats);
}

float *noodle_temp2_require(size_t required_floats) {
  return noodle_temp_require_impl(&temp_buff2, &temp_buff2_capacity, required_floats);
}

void noodle_temp_buffers_free(void) {
  if (temp_buff1 && temp_buff1_capacity != NOODLE_TEMP_EXTERNAL_CAPACITY_UNKNOWN) {
    free(temp_buff1);
  }
  if (temp_buff2 && temp_buff2_capacity != NOODLE_TEMP_EXTERNAL_CAPACITY_UNKNOWN) {
    free(temp_buff2);
  }
  temp_buff1 = NULL;
  temp_buff2 = NULL;
  temp_buff1_capacity = 0;
  temp_buff2_capacity = 0;
}

float *noodle_create_buffer(uint16_t size) {
  return (float *)malloc(size);
}

void noodle_delete_buffer(float *buffer) {
  free(buffer);
}

float *noodle_slice(float *flat,
                    size_t W,
                    size_t z) {
  return flat + z * W * W;
}

void noodle_setup_temp_buffers(void *b1,
                               void *b2) {
  // Legacy manual temp buffers. Capacity is unknown because the public
  // prototype is intentionally unchanged. Noodle will use these pointers
  // as-is and will not resize/free them automatically.
  temp_buff1 = b1;
  temp_buff2 = b2;
  temp_buff1_capacity = NOODLE_TEMP_EXTERNAL_CAPACITY_UNKNOWN;
  temp_buff2_capacity = NOODLE_TEMP_EXTERNAL_CAPACITY_UNKNOWN;
}

void noodle_setup_temp_buffers(void *b2) {
  temp_buff2 = b2;
  temp_buff2_capacity = NOODLE_TEMP_EXTERNAL_CAPACITY_UNKNOWN;
}
