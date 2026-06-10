/**
 * @file noodle_memory.cpp
 * @brief Noodle memory access management helpers.
 */
#include "noodle_internal.h"

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
  temp_buff1 = b1;
  temp_buff2 = b2;
}

void noodle_setup_temp_buffers(void *b2) {
  temp_buff2 = b2;
}
