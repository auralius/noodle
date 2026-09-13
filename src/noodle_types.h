/**
 * @file noodle_types.h
 * @brief Compile-time numeric types shared by Noodle buffers and tensors.
 */
#pragma once

#include <stdint.h>
#include "noodle_config.h"

#if defined(NOODLE_USE_INT8)
typedef int8_t NoodleData;
typedef int8_t NoodleWeight;
typedef int32_t NoodleBias;
typedef int32_t NoodleAccum;
#else
typedef float NoodleData;
typedef float NoodleWeight;
typedef float NoodleBias;
typedef float NoodleAccum;
#endif
