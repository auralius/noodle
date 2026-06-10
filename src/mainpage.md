# Noodle {#mainpage}

Noodle is a lightweight CNN-style inference library designed for Arduino-class
microcontrollers and other memory-constrained systems. Its primary design
principle is streaming-based execution: instead of storing all intermediate
tensors in RAM, Noodle can read inputs, weights, and biases from external
storage and write intermediate activations back to storage. This approach keeps
the peak memory footprint small and predictable.

This documentation describes the public API, the refactored implementation
layout, and the core invariants needed to use Noodle correctly and safely:
data layouts, file formats, filesystem selection, parameter layout, buffer
ownership, output sizing, and scratch-buffer management.

## What Noodle Is And Is Not

Noodle is:

- A compact set of C/C++ functions for convolution, activation, pooling,
  flattening, fully connected, and related CNN-style pipelines.
- Designed for memory-constrained environments, with APIs that support
  file-backed I/O to avoid large `W * W * C` allocations.
- File-backed, memory-backed, PROGMEM-backed, and mixed file/memory overloads
  where the layer family supports them.
- Backend-agnostic at the call site: filesystem operations are routed through a
  small abstraction layer. See @ref noodle_fs "Filesystem Backend Layer".
- A library of explicit layer primitives, including 2D convolution, 1D
  convolution, 2D depthwise convolution, memory-backed 2D transpose convolution,
  pooling, flattening, reshaping, global average pooling, global max pooling,
  fully connected layers, ReLU, sigmoid, softmax, rank-specific batch
  normalization, and max-index helpers.

Noodle is not:

- A training framework; it does not provide automatic differentiation or
  optimizers.
- A dynamic tensor runtime with graph scheduling.
- A replacement for highly optimized vendor DSP or accelerator libraries. Its
  focus is clarity, portability, and pedagogical transparency.

Model training and graph design happen outside the library; Noodle runs the
explicit pipeline you write in firmware.

## Key Design Concepts

### Streaming-First Memory Model

A common failure mode in embedded machine learning is that models work in
desktop environments but exceed the memory limits of microcontrollers. This is
largely due to tensor sizes scaling as `W * W * C`.

Noodle addresses this issue through a streaming execution model:

- Inputs may reside on external storage.
- Each layer consumes an input tensor from memory or a file.
- The output can be written to a new file.
- Only small working buffers are kept in RAM.

This design explains why multiple function variants exist, such as file-to-file,
memory-to-file, file-to-memory, and memory-to-memory operations.

### Public Include And Refactored Source Layout

The application-facing boundary remains simple: user sketches include
`noodle.h`. The refactor split the implementation into smaller files so each
piece of behavior has a clearer home, but it did not change the public include
model.

Public files:

- `noodle.h`: application-facing structs, layer calls, helpers, and Doxygen
  API documentation.
- `noodle_buffer.h`: grow-only Noodle-owned float buffers used by buffer-based
  convolution overloads.
- `noodle_config.h`: compile-time backend, file-format, pooling, and scratch
  sizing macros.
- `noodle_fs.h`: filesystem backend abstraction, path normalization, file
  handle type, and open/remove/seek helpers.

Implementation files:

- `noodle_io.cpp`: filesystem initialization and scalar/tensor file I/O.
- `noodle_memory.cpp`: raw buffer helpers, slicing, and global convolution
  scratch-buffer management.
- `noodle_buffer.cpp`: grow-only `NoodleBuffer` allocation helpers.
- `noodle_conv.cpp`: public Conv1D, Conv2D, Conv2DTranspose, and PROGMEM Conv2D
  wrappers, including `NoodleBuffer` wrappers for 2D/transpose convolution.
- `noodle_dw.cpp`: public depthwise-convolution wrappers, including
  `NoodleBuffer` wrappers.
- `noodle_fcn.cpp`: dense/fully connected overloads, including file-backed,
  memory-backed, and PROGMEM-backed parameter paths.
- `noodle_shape.cpp`: flatten, reshape, global average pooling, and global max
  pooling helpers.
- `noodle_math.cpp`: dot products, activations, max search, rank-specific batch
  normalization, and backward-compatible BN aliases.
- `noodle_internal.h` and `noodle_internal.cpp`: private shared declarations,
  global scratch-buffer state, low-level convolution/pooling kernels, shape
  formulas, raw tensor/activation helpers, and implementation helpers.

Application code should not include `noodle_internal.h`; it is only for Noodle's
implementation files.

### Data Layout Conventions

Noodle uses explicit and consistent layout conventions to maintain predictability
across platforms.

CHW file and tensor layout:

- Channel planes are stored sequentially: `[ch0 plane][ch1 plane]...[chC-1 plane]`.
- A 2D feature map is stored as `[C][W][W]`.

HWC-like flatten layout:

- `noodle_flat()` writes pixel-major interleaved channels.
- The output index is `out[(y * W + x) * C + c]`.
- `noodle_reshape()` converts that HWC-like order back to packed `[C][W][W]`
  channel-first order for square tensors.

When mixing streaming and in-memory operations, make sure each step expects the
layout produced by the previous step.

### Noodle And TFLite Micro Mental Models

Noodle is built around explicit streaming:

- Weights can live as files.
- Activations can live in memory or files.
- Layers are manually called one by one.
- Larger models can stream from FFat, SD, SD_MMC, or another selected backend.

In this model, inputs, weights, and biases may reside on external storage, and
only small working buffers need to stay in RAM. The application controls the
layer order, file names, intermediate tensors, and memory reuse.

TFLite Micro uses a different execution model:

- The model is represented as a TensorFlow Lite FlatBuffer.
- A `MicroInterpreter` is constructed with the model, an operator resolver, and
  a preallocated tensor arena.
- Tensor allocation prepares pointers for model inputs, outputs, intermediate
  tensors, and operator scratch data.
- Operators run through the interpreter and expect tensor data to be addressable
  through memory pointers.

This is the main design difference: Noodle exposes layer calls and file
streaming directly, while TFLite Micro centers execution around an interpreter,
a FlatBuffer model, and a tensor arena.

## Compile-Time Configuration

Configuration lives in `noodle_config.h`.

Select exactly one filesystem backend before including `noodle.h`:

- `NOODLE_USE_SDFAT`
- `NOODLE_USE_SD_MMC`
- `NOODLE_USE_FFAT`
- `NOODLE_USE_LITTLEFS`
- `NOODLE_USE_NONE`

If no backend macro is selected, `noodle_config.h` defaults to
`NOODLE_USE_SDFAT`.

File-backed scalar I/O is selected with `NOODLE_FILE_FORMAT`:

- `NOODLE_FILE_FORMAT_BIN` is the default. Floats are raw little-endian IEEE-754
  float32 values; bytes are raw `uint8_t` values.
- `NOODLE_FILE_FORMAT_TEXT` stores one ASCII numeric value per line.

Filename extensions are conventions only. Noodle reads and writes according to
`NOODLE_FILE_FORMAT`, so a `.bin` file must be used with binary mode and a
`.txt` file must be used with text mode.

2D pooling behavior is selected with `NOODLE_POOL_MODE`:

- `NOODLE_POOL_NONE`
- `NOODLE_POOL_MAX`
- `NOODLE_POOL_MEAN`

The 1D pooling helper performs max pooling.

Fully connected file reads can be tuned with `NOODLE_FCN_BLOCK`. Float-input
fully connected layers with `FCNFile` parameters read weights in chunks of up to
that many floats. In binary mode the chunk is read as raw float32 data; in text
mode the same chunk is filled by parsing one scalar at a time. The default is
128 floats, which uses `NOODLE_FCN_BLOCK * sizeof(float)` bytes of stack for the
weight buffer.

Convolution and depthwise-convolution paths copy one kernel into stack scratch
using `NOODLE_MAX_K`. Set it to the largest kernel width used by the firmware;
the default is 5.

## Tensor And Parameter Layouts

Unless an API says otherwise, Noodle uses channel-first packed storage.

Feature maps:

- 2D tensors: `[C][W][W]`
- 1D tensors: `[C][W]`

Layer parameters:

- 2D convolution weights: `[O][I][K][K]`
- 1D convolution weights: `[O][I][K]`
- 2D depthwise weights: `[C][K][K]`
- 2D transpose convolution weights: `[O][I][K][K]`
- Fully connected weights: `[O][I]`
- Bias files or arrays: one scalar per output channel or neuron
- Batch normalization arrays: `[gamma[N]][beta[N]][mean[N]][var[N]]`, where
  `N` is a vector length for BN1D or a channel count for BN2D

File-backed fully connected layers consume weights sequentially in `[O][I]`
order. The float-input variants use block reads for the weight stream, but the
file layout does not change.

PROGMEM-backed convolution and fully connected parameter structs use the same
packed layouts as their memory-backed equivalents.

### Conv2DTranspose Output Sizing

Noodle's transpose-convolution API is memory-backed and uses the same
`[O][I][K][K]` kernel layout as normal Conv2D. The exporter can convert the
kernel layout, but the firmware still chooses the layer geometry through `K`,
`P`, `S`, and `OP`.

For explicit symmetric padding, `OP` is user-computed from the desired output
width:

@code{.cpp}
V = (W - 1) * S - 2 * P + K + OP
@endcode

Equivalently, after choosing the desired `V`, solve:

@code{.cpp}
OP = V - ((W - 1) * S - 2 * P + K)
@endcode

`OP` should normally be in the range `[0, S - 1]`, matching common
transpose-convolution conventions. When `P = 65535`, Noodle uses
Keras/TF-style SAME sizing for transpose convolution: `V = W * S`, and the
helper derives the asymmetric crop internally.

`noodle_flat()` is the main layout conversion helper. It reads packed
`[C][V][V]` input and writes HWC-like spatial-major output:

@code{.cpp}
output[pixel * C + channel]
@endcode

`noodle_reshape()` performs the inverse memory layout conversion for square
tensors. It reads HWC-like spatial-major input and writes packed channel-first
output:

@code{.cpp}
dst_chw[channel * W * W + pixel] = src_hwc[pixel * C + channel]
@endcode

Batch normalization is explicit about tensor rank:

- `noodle_bn1d()` and `noodle_bn1d_relu()` operate on packed `[N]` vectors after
  FCN/Dense layers, GAP/GMP, or flatten-like outputs.
- `noodle_bn2d()` and `noodle_bn2d_relu()` operate on packed `[C][W][W]`
  channel-first tensors after Conv2D, depthwise Conv2D, or pointwise Conv2D.
- `noodle_bn()` and `noodle_bn_relu()` remain as backward-compatible aliases for
  the old 2D channel-first behavior.

## Buffers And Scratch Space

Noodle has two different buffer concepts:

- Internal convolution scratch buffers are managed globally by Noodle.
- Application tensor buffers are either caller-owned raw `float *` arrays or
  Noodle-owned `NoodleBuffer` descriptors.

Convolution and depthwise-convolution overloads now request their internal
scratch space on demand. Noodle allocates the global scratch buffers when first
needed, grows them when a larger layer requires more space, and reuses them for
later calls. Call `noodle_temp_buffers_free()` when the application wants to
release automatically allocated scratch memory.

The legacy `noodle_setup_temp_buffers()` calls still exist for sketches that
want to provide fixed external scratch buffers. When those pointers are
installed, Noodle uses them as-is and does not resize or free them because their
capacity is unknown. Make sure manually installed buffers are large enough for
every layer that will use them.

Common internal scratch roles:

- Temp 1: input scratch for file-backed input planes or sequences.
- Temp 2: accumulation scratch for a pre-pooling output plane or sequence.

For application tensors, raw-pointer overloads expect the caller to provide
correctly sized input and output arrays. `NoodleBuffer` overloads are the
grow-only alternative: initialize a descriptor with `noodle_buffer_init()`, pass
it to supported convolution calls, and free it with `noodle_buffer_free()` when
the tensor storage is no longer needed. A `NoodleBuffer` owns only memory
allocated by Noodle and tracks capacity in float elements.

## Quick Start

### 1. Select A Filesystem Backend

Enable a backend using build flags or preprocessor definitions before including
`noodle.h`. The exact build flag syntax depends on your platform or build
system.

- SD via SdFat: `NOODLE_USE_SDFAT`
- SD_MMC: `NOODLE_USE_SD_MMC`
- FFat: `NOODLE_USE_FFAT`
- LittleFS: `NOODLE_USE_LITTLEFS`
- No external file storage: `NOODLE_USE_NONE`

If no backend is selected, `noodle_config.h` defaults to `NOODLE_USE_SDFAT`.
See @ref noodle_fs "Filesystem Backend Layer".

Select the scalar file format at compile time as well:

@code{.cpp}
#define NOODLE_USE_FFAT
#define NOODLE_FILE_FORMAT NOODLE_FILE_FORMAT_BIN
#include "noodle.h"
@endcode

Initialize storage when a real filesystem backend is used:

@code{.cpp}
if (!noodle_fs_init()) {
  // Handle storage initialization failure.
}
@endcode

### 2. Prepare Tensor Storage

Noodle allocates convolution scratch space automatically. For layer inputs and
outputs, either provide raw arrays with enough room for the expected tensor
shape, or use `NoodleBuffer` for grow-only Noodle-owned tensor storage.

@code{.cpp}
NoodleBuffer feat_a;
NoodleBuffer feat_b;

noodle_buffer_init(&feat_a);
noodle_buffer_init(&feat_b);

float *input = noodle_buffer_require(&feat_a, W * W);
// Fill input with one packed [1][W][W] feature map.
// Later, pass &feat_a and &feat_b to NoodleBuffer convolution overloads.

// At shutdown or when the tensors are no longer needed:
noodle_buffer_free(&feat_a);
noodle_buffer_free(&feat_b);
noodle_temp_buffers_free();
@endcode

The global scratch buffers and shared file handles mean Noodle calls are not
re-entrant. In RTOS-based systems, treat a Noodle pipeline as a single-threaded
worker unless you add external serialization.

### 3. Construct A Processing Pipeline

Create a small explicit pipeline:

@code{.cpp}
Conv conv1;
conv1.K = 3;
conv1.P = 1;
conv1.S = 1;
conv1.weight_fn = "w01.bin";
conv1.bias_fn = "b01.bin";
conv1.act = ACT_RELU;

Pool pool1;
pool1.M = 2;
pool1.T = 2;

uint16_t V = noodle_conv_float("input.bin", 1, 8, "c1.bin", W, conv1, pool1, NULL);

float flat[8 * V * V];
noodle_flat("c1.bin", flat, V, 8);

float scores[10];
FCNFile head;
head.weight_fn = "w02.bin";
head.bias_fn = "b02.bin";
head.act = ACT_SOFTMAX;

noodle_fcn(flat, 8 * V * V, 10, scores, head, NULL);
@endcode

External storage acts as activation memory in streaming pipelines. For models
that fit in RAM, use the memory-backed overloads and `.h` arrays instead.

## Export Workflow

Noodle is inference-only. Train with Keras, PyTorch, or another desktop
framework, then export parameters into Noodle's layouts.

The included `model_exporter.py` writes text scalar files and matching C/C++
headers:

- `.h` files contain C arrays for memory-backed inference.
- `.txt` files contain one scalar per line for text-mode file-backed inference.

Use `txt2bin.py` to convert `.txt` float parameter files to `.bin` when
`NOODLE_FILE_FORMAT_BIN` is selected. Each file is still read in the same Noodle
layout; only the scalar encoding changes.

The exporter handles the common layouts used by this repository:

- Conv2D: Keras `(Kh, Kw, Cin, Cout)` to Noodle `[O][I][K][K]`
- DepthwiseConv2D with multiplier 1: Keras `(Kh, Kw, Cin, 1)` to Noodle
  `[C][K][K]`
- Conv1D: Keras `(K, Cin, Cout)` to Noodle `[O][I][K]`
- Dense: Keras `(Din, Dout)` to Noodle `[O][I]`
- Batch normalization: `[gamma][beta][mean][var]`

For models with Conv2DTranspose, prefer `exporter_model(model, out_dir)` so the
exporter can distinguish transpose convolution kernels from regular Conv2D
kernels. Keras Conv2DTranspose uses `(Kh, Kw, Cout, Cin)` and is exported to
Noodle `[O][I][K][K]` without a spatial kernel flip. The exporter does not
choose `OP`; set `conv.OP` in firmware from the intended transpose-convolution
output size.

## Documentation Map

The generated reference is organized around:

- @ref noodle_public "Public API": application-facing structs, functions, and helpers.
- @ref noodle_fs "Filesystem Backend Layer": filesystem backend types, path
  normalization, and file helpers.
- @ref noodle_internal "Internal Helpers": lower-level routines documented for
  maintainers.
- @ref noodle_api "Implementation Files": source-level implementation
  documentation.

Configuration macros are described in `noodle_config.h`. When reading the
implementation, start from the public declaration in `noodle.h`, then move to
the matching refactored `.cpp` file; shared private pieces are declared in
`noodle_internal.h`.

## Limitations And Notes

- Several helpers rely on shared file handles and shared global scratch buffers.
  Treat the library as non-reentrant unless you add external serialization.
- `noodle_internal.h` is private implementation surface and may change without
  preserving application-level compatibility.
- Path normalization for slash-requiring filesystems uses a static buffer.
- Many 2D APIs assume square feature maps (`W * W`) rather than rectangular
  tensors.
- `NOODLE_MAX_K` must be large enough for the largest convolution kernel copied
  into stack scratch.
- Explicit convolution padding uses one padding value per side. Passing
  `P = 65535` requests SAME-style output sizing, and the helper resolves the
  required top/left and bottom/right padding internally.
- For transpose convolution with explicit padding, `OP` is an output-size choice
  made by the caller: `V = (W - 1) * S - 2 * P + K + OP`.
- Pooling uses valid regions only; pooling layers do not add padding.
- File-backed execution can be I/O-bound on small boards. Binary scalar files
  are usually faster and smaller than text scalar files.

## Glossary

- CHW: channel-first planar storage.
- HWC-like flatten: spatial-major data with channels interleaved per pixel.
- Streaming: moving tensors through external storage to reduce peak RAM usage.
- `I` / `O`: input and output channel or neuron counts.
- `W`: input width for square 2D maps, or sequence length for 1D maps.
- `K`: kernel size.
- `P`: padding.
- `S`: stride.
- `OP`: user-computed output padding for transpose convolution.
