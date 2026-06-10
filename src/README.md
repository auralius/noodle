# Noodle Refactored Layout

This refactor keeps `noodle.h` as the public user-facing include, while splitting
implementation into a small number of maintainable modules.

## Public files

| File | Purpose |
|---|---|
| `noodle.h` | Main public API. User sketches should include this. |
| `noodle_buffer.h` | Public grow-only tensor buffer helper API. |
| `noodle_config.h` | Compile-time configuration. |
| `noodle_fs.h` | Filesystem backend abstraction/configuration. |

## Implementation files

| File | Purpose |
|---|---|
| `noodle_internal.h` | Private declarations shared by implementation files. Not intended for user sketches. |
| `noodle_internal.cpp` | Private shared globals and helper setup. |
| `noodle_io.cpp` | File/storage I/O helpers. |
| `noodle_math.cpp` | Reusable math primitives: dot product, max search, BN1D, BN2D, and BN+ReLU helpers. |
| `noodle_conv.cpp` | Outer Conv1D, Conv2D, ConvTranspose2D, PROGMEM Conv2D, and NoodleBuffer Conv2D wrappers. |
| `noodle_fcn.cpp` | Dense/FCN layers, including file-backed, memory-backed, and PROGMEM-backed parameter paths. |
| `noodle_shape.cpp` | Flatten, reshape, global average pooling, and global max pooling. |
| `noodle_dw.cpp` | Depthwise convolution and NoodleBuffer depthwise wrappers. |
| `noodle_memory.cpp` | Raw buffer helpers, slicing, and global convolution scratch-buffer management. |
| `noodle_buffer.cpp` | Grow-only NoodleBuffer allocation helpers. |

## User include

Existing examples should continue to use:

```cpp
#include "noodle.h"
```

Do not include `noodle_internal.h` from sketches. It is for the implementation
files only.
