# Unified Float32 / INT8 Noodle

Noodle remains one library. Float32 is the default, so existing sketches and
parameter files continue to build without defining a numerical-mode macro.

```cpp
#include <noodle.h>                 // original Float32 behavior
```

Full integer inference is selected at compile time:

```cpp
#define NOODLE_USE_INT8
#include <noodle.h>
```

For PlatformIO, prefer a project-wide build flag so every translation unit sees
the same ABI:

```ini
build_flags =
  -D NOODLE_USE_INT8
  -D NOODLE_USE_FFAT
```

Do not define `NOODLE_USE_INT8` in only one `.cpp` file while compiling the
library without it. The macro changes `NoodleData`, layer structures, and tensor
metadata and therefore must be consistent for the entire firmware.

## Numeric contract

| Item | Float32 mode | INT8 mode |
|---|---|---|
| Activation tensor | `float` | `int8_t` |
| Weight | `float` (or legacy q8 storage) | `int8_t` |
| Bias | `float` | `int32_t` |
| Accumulator | `float` | integer MAC, saturated to `int32_t` before requantization |
| Output | `float` | `int8_t` |

INT8 tensors follow the affine convention

```
real = (q - zero_point) * scale
```

Weighted layers additionally receive per-output-channel `multiplier` and
`shift` arrays derived from

```
input_scale * weight_scale[channel] / output_scale
```

The intended exporter is a full-INT8 TFLite/LiteRT model. Noodle trusts the
TFLite scales, zero points, int8 weights, and int32 biases; the host exporter
transposes weights to Noodle's CHW layouts and generates multiplier/shift files.

## Backends

The existing filesystem selection remains unchanged: SdFat, SD_MMC, FFat,
LittleFS, or no filesystem. INT8 weighted operators support memory-backed and
file-backed parameters. Near AVR PROGMEM is supported for Conv/Depthwise, and
far AVR PROGMEM addresses are supported for FCN parameters.

File-backed INT8 weighted layers use four files:

- weight: signed int8
- bias: signed int32
- multiplier: signed int32 per output channel
- shift: signed int32 per output channel

## Compatibility boundary

The newer `NoodleTensor` API is available in both modes with the same layer
function names. The old Float32 raw-pointer, file-to-file activation, BatchNorm,
and legacy `noodle_conv_float(...)` overloads remain in Float32 mode. BatchNorm
should be folded by the TFLite converter for INT8 deployment.

Max pool, reshape, flatten, GMP, and concatenation preserve quantization. Mean
pool preserves the input tensor quantization. Conv, Depthwise, Conv1D,
ConvTranspose, FCN, GAP, sigmoid, and softmax perform integer output conversion.

## Input setup

```cpp
NoodleTensor input;
noodle_tensor_init(&input);
noodle_tensor_set_quantization(&input, input_scale, input_zero_point);
int8_t *data = noodle_tensor_require_2d(&input, channels, width);
```

Output scale and zero point are taken from the selected layer descriptor.
