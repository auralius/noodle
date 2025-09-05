<p align="center">
  <img src="./noodle.png" alt="Description" width="100"> 
</p>

<h1 align="center">Noodle 🍜≈🧠
  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16239228.svg)](https://doi.org/10.5281/zenodo.16239228)
  
</h1>


**Noodle** is a lightweight convolutional neural network inference library designed for MCUs with **very small RAM**.  
It streams activations and weights from **SD/FFat/SD_MMC** filesystems to overcome RAM limitations, while providing modular primitives for **convolution, pooling, FCN, and activations**. During the development, we typically test Noodle with Arduino UNO R3, UNO R4, and some ESP32 variants.

---

## Configuration

Configuration is controlled in [`noodle_config.h`](./noodle_config.h):

- **Filesystem Backend** (select exactly one):
  - `NOODLE_USE_SDFAT`
  - `NOODLE_USE_SD_MMC`
  - `NOODLE_USE_FFAT`

- **Pooling Mode**:
  ```c
  #define NOODLE_POOL_MAX   1
  #define NOODLE_POOL_MEAN  2
  #define NOODLE_POOL_MODE NOODLE_POOL_MEAN  // or NOODLE_POOL_MAX
  ```

See [`noodle_fs.h`](./noodle_fs.h) for backend definitions.

---

## Data Structures

### `enum Activation`
- `ACT_NONE` — no activation  
- `ACT_RELU` — ReLU clamp  
- `ACT_SOFTMAX` — softmax (applied at vector level)

### `struct Conv`
Parameters for convolution:
- `K` — kernel size  
- `P` — padding  
- `S` — stride  
- `weight_fn` — filename template for weights (per input/output channel)  
- `bias_fn` — filename of biases (one per output channel)  
- `act` — activation (see `Activation`)

### `struct Pool`
Parameters for pooling:
- `M` — kernel size  
- `T` — stride

### `struct FCNFile`
- `weight_fn` — filename of weights  
- `bias_fn` — filename of biases  
- `act` — activation

### `struct FCNMem`
In-memory FCN layer parameters:
- `weight` — pointer to row-major weights `[n_outputs, n_inputs]`  
- `bias` — pointer to bias array  
- `act` — activation

---

## Utilities

### Memory
- `float* noodle_slice(float* flat, size_t W, size_t z)`  
  Returns pointer to the `z`-th slice of a stacked `[Z, W, W]` tensor.  

- `void noodle_setup_temp_buffers(void *b1, void *b2)`  
  Registers two scratch buffers (input + accumulator) for streamed ops.  

- `float* noodle_create_buffer(uint16_t size)` / `void noodle_delete_buffer(float* buffer)`  
  Allocate/free float buffers.  

- `void noodle_reset_buffer(float* buffer, uint16_t n)`  
  Zero-fill a float buffer.  

### File/FS Helpers
- `NDL_File noodle_open_file_for_write(const char* fn)`  
- `size_t noodle_read_bytes_until(NDL_File&, char terminator, char* buf, size_t len)`  
- `bool noodle_sd_init()` / `bool noodle_sd_init(int clk, int cmd, int d0)`  
- `void noodle_n2ll(uint16_t number, char* out)` — encodes int → 2-letter code (aa..zz)  
- `void noodle_delete_file(const char* fn)`  

### Scalar I/O
- `void noodle_write_float(NDL_File&, float d)` / `float noodle_read_float(NDL_File&)`  
- `void noodle_write_byte(NDL_File&, byte d)` / `byte noodle_read_byte(NDL_File&)`  

### Array/Grid I/O
- `void noodle_array_to_file(float* arr, const char* fn, uint16_t n)`  
- `void noodle_grid_to_file(byte*/float* grid, const char* fn, uint16_t n)`  
- `void noodle_array_from_file(const char* fn, float* buf, uint16_t K)`  
- `void noodle_grid_from_file(const char* fn, byte*/int8_t*/float* buf, uint16_t K)`  

---

## Core Ops

### Convolution (2D)

#### Primitives
- `uint16_t noodle_do_conv(byte* grid, float* kernel, K, W, float* output, P, S)`  
- `uint16_t noodle_do_conv(float* grid, float* kernel, K, W, float* output, P, S)`  

Output size: `V = (W - K + 2P)/S + 1`.

#### File/Memory Variants
- `noodle_conv_byte` — File→File, byte inputs  
- `noodle_conv_float` — Overloaded for:
  - File→File
  - File→Memory
  - Memory→File
  - Memory→Memory

Each variant:
- Iterates input channels **I**, output channels **O**  
- Loads weights/bias  
- Applies conv + bias + activation  
- Writes pooled feature maps to file or memory  

### Pooling
- `noodle_do_pooling(float* in, W, K, S, const char* fn)` → File output  
- `noodle_do_pooling(const float* in, W, K, S, float* out)` → Memory output  
- `noodle_do_pooling1d(float* in, W, K, S, const char* fn)` → 1D MAX pooling  

Output size: `Wo = (W - K)/S + 1`.

---

## Fully Connected Layers (FCN)

- `uint16_t noodle_fcn(const int8_t* in, n_inputs, n_outputs, const char* out_fn, const FCNFile&, CBFPtr cb)`  
- `uint16_t noodle_fcn(const float* in, n_inputs, n_outputs, const char* out_fn, const FCNFile&, CBFPtr cb)`  
- `uint16_t noodle_fcn(const byte* in, n_inputs, n_outputs, const char* out_fn, const FCNFile&, CBFPtr cb)`  
- `uint16_t noodle_fcn(const char* in_fn, n_inputs, n_outputs, const char* out_fn, const FCNFile&, CBFPtr cb)`  
- `uint16_t noodle_fcn(const float* in, n_inputs, n_outputs, float* out, const FCNMem&, CBFPtr cb)`  

General behavior:
- Computes `y = W·x + b`  
- Optional activation: ReLU, Softmax  
- Supports **File→File**, **File→Memory**, **Memory→File**, **Memory→Memory**  

---

## Flattening
- `uint16_t noodle_flat(const char* in_fn, float* out, uint16_t V, uint16_t n_filters)` — File→Memory  
- `uint16_t noodle_flat(float* in, float* out, uint16_t V, uint16_t n_filters)` — Memory→Memory  

Output: `V * V * n_filters` vector.

---

## Activations
- `uint16_t noodle_do_bias(float* out, float bias, uint16_t n)` — bias + ReLU  
- `uint16_t noodle_do_bias_act(float* out, float bias, uint16_t n, Activation act)` — bias + activation  
- `uint16_t noodle_soft_max(float* vec, uint16_t n)` — in-place softmax  
- `uint16_t noodle_sigmoid(float* vec, uint16_t n)` — in-place sigmoid  

---

## 📏 1D Convolution
- `uint16_t noodle_do_conv1d(float* in, float* kernel, W, K, float* out, P, S)`  
- `uint16_t noodle_conv1d(float* in, float* out, n_inputs, n_outputs, const char* in_fn, const char* out_fn, W, const Conv&, const Pool&, CBFPtr cb)` — with pooling  
- `uint16_t noodle_conv1d(float* in, float* out, n_inputs, n_outputs, const char* in_fn, const char* out_fn, W, const Conv&, CBFPtr cb)` — no pooling  

---

## Example Usage

```cpp
#include "noodle.h"

Conv conv = {3, 1, 1, "w____", "b.txt", ACT_RELU};
Pool pool = {2, 2};

void setup() {
  static byte buffer1[784];    // temp input
  static float buffer2[784];   // temp accumulator

  noodle_setup_temp_buffers(buffer1, buffer2);

  noodle_sd_init();  // init filesystem

  // Run conv layer
  noodle_conv_byte("in____", 1, 6, "out__", 28, conv, pool);
}
```

---

## References

- [Noodle source code](./noodle.cpp)  
- [Noodle headers](./noodle.h)  
- [Noodle config](./noodle_config.h)  
- [Filesystem backend](./noodle_fs.h)  


## Authors

- Auralius Manurung — Universitas Telkom, Bandung  
- Lisa Kristiana — ITENAS, Bandung

