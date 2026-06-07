# Noodle Refactored Layout

This refactor keeps `noodle.h` as the public user-facing include, while splitting
implementation into a small number of maintainable modules.

## Public files

| File | Purpose |
|---|---|
| `noodle.h` | Main public API. User sketches should include this. |
| `noodle_config.h` | Compile-time configuration. |
| `noodle_fs.h` | Filesystem backend abstraction/configuration. |

## Implementation files

| File | Purpose |
|---|---|
| `noodle_internal.h` | Private declarations shared by implementation files. Not intended for user sketches. |
| `noodle_internal.cpp` | Private shared globals and helper setup. |
| `noodle_io.cpp` | File/storage I/O helpers. |
| `noodle_math.cpp` | Reusable math primitives: dot product, max search, BN1D, BN2D, and BN+ReLU helpers. |
| `noodle_conv.cpp` | Outer Conv1D, Conv2D, and ConvTranspose2D layer APIs only. |
| `noodle_fcn.cpp` | Dense/FCN layers plus output activations. |
| `noodle_flat.cpp` | Flatten, reshape, global average pooling, and global max pooling. |
| `noodle_dw.cpp` | Depthwise convolution. |
| `noodle.cpp` | Compatibility placeholder for build systems that expect the file. |

## User include

Existing examples should continue to use:

```cpp
#include "noodle.h"
```

Do not include `noodle_internal.h` from sketches. It is for the implementation
files only.


## Layer-boundary refinement

`noodle_conv.cpp` now contains only the outer convolution layer APIs: Conv1D, Conv2D, and ConvTranspose2D wrappers. Private implementation details such as low-level convolution kernels, padded reads, pooling helpers, shape computation, and transpose scatter logic live in `noodle_internal.cpp` and are declared in `noodle_internal.h`. This keeps the convolution file readable and makes the user-facing layer boundary clearer.


## Batch-normalization naming

Batch normalization is now explicit about tensor rank:

| Function family | Input layout | Typical use |
|---|---|---|
| `noodle_bn1d()` / `noodle_bn1d_relu()` | `[N]` vector | after FCN/Dense, GAP, GMP, or flatten-like vector outputs |
| `noodle_bn2d()` / `noodle_bn2d_relu()` | `[C][W][W]` channel-first tensor | after Conv2D, DWConv, or pointwise Conv2D |

The old `noodle_bn()` and `noodle_bn_relu()` names are kept as backward-compatible aliases for the old 2D channel-first behavior. New code should prefer the explicit names. `noodle_unpack_bn_params()` is now private and declared only in `noodle_internal.h`.
