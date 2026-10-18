#pragma once

#include <stdint.h>
namespace noodle_export_i8
{
    static constexpr float layer01_input_scale = 1.0f;
    static constexpr float layer01_output_scale = 1.75215828f;
    static constexpr int32_t layer01_input_zero_point = -128;
    static constexpr int32_t layer01_output_zero_point = -128;
    static constexpr int32_t layer01_activation_min = -128;
    static constexpr int32_t layer01_activation_max = 127;
    static constexpr float layer02_input_scale = 1.75215828f;
    static constexpr float layer02_output_scale = 1.88650119f;
    static constexpr int32_t layer02_input_zero_point = -128;
    static constexpr int32_t layer02_output_zero_point = 12;
    static constexpr int32_t layer02_activation_min = -128;
    static constexpr int32_t layer02_activation_max = 127;
}
