# Brief API References


---
Tiny, file-streamed CNN/ML primitives for microcontrollers with *very* small RAM budgets.  

Noodle supports in-memory and file-system-streamed (SD/FFat/SD_MMC/LittleFS/SdFat) execution for:

- 2D/1D convolution (+ optional pooling)
- bias + activation
- flatten
- fully connected (FCN)
- softmax / sigmoid

This reference is generated from `noodle.h`, `noodle_config.h`, and `noodle_fs.h`.

---
## 1) Configuration

### 1.1 Filesystem backend selection (`noodle_config.h` / `noodle_fs.h`)

Define **exactly one** before including `noodle.h`:

- `NOODLE_USE_SDFAT`
- `NOODLE_USE_SD_MMC`
- `NOODLE_USE_FFAT`
- `NOODLE_USE_LITTLEFS`

`noodle_fs.h` exposes:

- `using NDL_File = ...`  
  - `FsFile` for SdFat, otherwise Arduino `File`
- `NOODLE_FS` handle  
  - `SdFat NOODLE_FS` (extern, defined in `noodle.cpp`) or the FS singleton (`FFat`, `SD_MMC`, `LittleFS`)
- Backend-neutral open/remove helpers:
  - `NDL_File noodle_fs_open_read(const char* path)`
  - `NDL_File noodle_fs_open_write(const char* path)`
  - `bool noodle_fs_remove(const char* path)`

### 1.2 Pooling mode (`noodle_config.h`)

Compile-time pooling mode for 2D pooling:

- `#define NOODLE_POOL_MAX  1`
- `#define NOODLE_POOL_MEAN 2`
- `#define NOODLE_POOL_MODE NOODLE_POOL_MAX` **or** `NOODLE_POOL_MEAN`

---
## 2) Naming convention for streamed channel files (two-letter indices)

### Convolution kernels (4D tensors)

**Filename format:**  
`w<NN><in><out>.txt`

Where:

- `<NN>` is a two-digit, zero-padded index identifying a **weight tensor**
- `<in>` is a two-letter code for the input-channel index (`aa`, `ab`, …, `zz`)
- `<out>` is a two-letter code for the output-channel index

**Example:**  
`w01aaab.txt`

Each file contains one flattened spatial kernel for a single `(input channel, output channel)` pair.

### Dense weights (2D tensors)

**Filename format:**  
`w<NN>.txt`

Where:

- `<NN>` is a two-digit, zero-padded index identifying a **weight tensor**

**Example:**  
`w03.txt`

Dense weights are exported as a transposed, flattened array, one value per line.

### Bias vectors (1D tensors)

**Filename format:**  
`b<NN>.txt`

Where:

- `<NN>` is a two-digit, zero-padded index identifying a **bias vector**

**Example:** 
`b02.txt`

Each file contains flattened bias values, one value per line.

---
## 3) Types

### 3.1 Activation
```cpp
enum Activation : uint8_t { ACT_NONE = 0, ACT_RELU = 1, ACT_SOFTMAX = 2 };
```

### 3.2 `Conv` (shared params for 1D/2D)
```cpp
struct Conv {
  uint16_t K;              // kernel size (2D: KxK, 1D: K)
  uint16_t P = 0;          // padding
  uint16_t S = 1;          // stride
  const char *weight_fn;   // file template for weights
  const char *bias_fn;     // bias filename
  Activation act = ACT_RELU;
};
```

### 3.3 `Pool`
```cpp
struct Pool {
  uint16_t M = 2;  // pool kernel
  uint16_t T = 2;  // pool stride
};
```

### 3.4 Progress callback
```cpp
typedef void (*CBFPtr)(float progress);  // progress in [0,1]
```

### 3.5 FCN parameter blocks
```cpp
struct FCN     
{ 
  const char *weight_fn; 
  const char *bias_fn; 
  Activation act = ACT_RELU; 
};

struct FCNFile 
{ 
  const char *weight_fn; 
  const char *bias_fn; 
  Activation act = ACT_RELU; 
};

struct FCNMem {
  const float *weight;  // row-major [n_outputs, n_inputs]
  const float *bias;    // length n_outputs
  Activation act = ACT_RELU;
};
```

---
## 4) Initialization & scratch buffers

### 4.1 Temporary buffers (required for some streamed ops)
```cpp
void noodle_setup_temp_buffers(void *b1, void *b2 = NULL);
```

Provides two reusable scratch buffers used internally by file-streaming convolution/FCN variants.

Typical guidance (from header docs):
- temp buffer #1: ≥ `W*W*sizeof(input_element)`
- temp buffer #2: ≥ `W*W*sizeof(float)` (accumulator)

---
## 5) Filesystem utilities

### 5.1 Open for write (delete first)
```cpp
NDL_File noodle_open_file_for_write(const char* fn);
```

### 5.2 Read bytes until terminator
```cpp
size_t noodle_read_bytes_until(NDL_File &file, char terminator, char *buffer, size_t length);
```
Reads into `buffer` (always NUL-terminated), stops at `terminator` or `length-1`.

### 5.3 Backend init
```cpp
bool noodle_sd_init(int clk_pin, int cmd_pin, int d0_pin);
bool noodle_sd_init();
```
Pins variant is meaningful for SD_MMC; default form uses backend defaults.

### 5.4 Delete file
```cpp
void noodle_delete_file(const char *fn);
```

---
## 6) Scalar I/O helpers (text files)

```cpp
void  noodle_write_float(NDL_File &f, float d);
float noodle_read_float(NDL_File &f);

byte  noodle_read_byte(NDL_File &f);
void  noodle_write_byte(NDL_File &f, byte d);
```

---
## 7) Memory helpers

```cpp
float* noodle_create_buffer(uint16_t size); 
void   noodle_delete_buffer(float *buffer);
void   noodle_reset_buffer(float *buffer, uint16_t n); 
```

---

# 8) Array / grid I/O (text files)

### 8.1 Write
```cpp
void noodle_array_to_file(float *array, const char *fn, uint16_t n);
void noodle_grid_to_file(byte  *grid,  const char *fn, uint16_t n); /
void noodle_grid_to_file(float *grid,  const char *fn, uint16_t n);
```

### 8.2 Read
```cpp
void noodle_array_from_file(const char *fn, float *buffer, uint16_t K);

void noodle_grid_from_file(const char *fn, byte   *buffer, uint16_t K);
void noodle_grid_from_file(const char *fn, int8_t *buffer, uint16_t K);
void noodle_grid_from_file(const char *fn, float  *buffer, uint16_t K);
```

### 8.3 Padded accessors
```cpp
float noodle_get_padded_x(byte  *grid, int16_t i, int16_t j, int16_t W, int16_t P);
float noodle_get_padded_x(float *grid, int16_t i, int16_t j, int16_t W, int16_t P);
```
Returns 0 outside bounds, otherwise grid value as float.

---
## 9) Convolution & pooling primitives

### 9.1 2D convolution (in-memory primitive)
Output spatial size: `V = (W - K + 2P)/S + 1`

```cpp
uint16_t noodle_do_conv(byte  *grid, float *kernel, uint16_t K, uint16_t W,
                        float *output, uint16_t P, uint16_t S);

uint16_t noodle_do_conv(float *grid, float *kernel, uint16_t K, uint16_t W,
                        float *output, uint16_t P, uint16_t S);
```

### 9.2 Bias (+ optional activation)
```cpp
uint16_t noodle_do_bias(float *output, float bias, uint16_t n);  // legacy: bias + ReLU

uint16_t noodle_do_bias_act(float *output, float bias, uint16_t n, Activation act);
```

### 9.3 2D pooling (mode selected by `NOODLE_POOL_MODE`)
Output spatial size: `V_out = (V - K)/S + 1`

```cpp
uint16_t noodle_do_pooling(float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);
uint16_t noodle_do_pooling(const float *input, uint16_t W, uint16_t K, uint16_t S, float *output);
```

### 9.4 1D pooling (file output)
```cpp
uint16_t noodle_do_pooling1d(float *input, uint16_t W, uint16_t K, uint16_t S, const char *fn);
```

### 9.5 1D convolution primitive
Output length: `V = (W - K + 2P)/S + 1`

```cpp
uint16_t noodle_do_conv1d(float *input, float *kernel, uint16_t W, uint16_t K,
                          float *output, uint16_t P, uint16_t S);
```

---
## 10) Streamed Convolution APIs

These functions use the filename tokenization convention (two-letter indices) where applicable.

### 10.1 2D convolution (+ optional pooling)

**File → File (BYTE input feature maps)**
```cpp
uint16_t noodle_conv_byte(const char *in_fn,
                          uint16_t n_inputs,
                          uint16_t n_outputs,
                          const char *out_fn,
                          uint16_t W,
                          const Conv &conv,
                          const Pool &pool,
                          CBFPtr progress_cb = NULL);
```

**File → File (FLOAT input feature maps)**
```cpp
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);
```

**File → Memory (FLOAT inputs, output tensor `[O, Wo, Wo]`)**
```cpp
uint16_t noodle_conv_float(const char *in_fn,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);
```

**Memory → File (FLOAT inputs)**
```cpp
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           const char *out_fn,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);
```

**Memory → Memory (FLOAT inputs)**
```cpp
uint16_t noodle_conv_float(float *input,
                           uint16_t n_inputs,
                           uint16_t n_outputs,
                           float *output,
                           uint16_t W,
                           const Conv &conv,
                           const Pool &pool,
                           CBFPtr progress_cb = NULL);
```

Return value is the output spatial size after pooling (if enabled by your `Pool` settings).

### 10.2 1D convolution (streamed, tokenized filenames)

**File → File, with pooling**
```cpp
uint16_t noodle_conv1d(float *input, float *output,
                       uint16_t n_inputs, uint16_t n_outputs,
                       const char *in_fn, const char *out_fn,
                       uint16_t W, const Conv &conv, const Pool &pool,
                       CBFPtr progress_cb = NULL);
```

**File → File, no pooling**
```cpp
uint16_t noodle_conv1d(float *input, float *output,
                       uint16_t n_inputs, uint16_t n_outputs,
                       const char *in_fn, const char *out_fn,
                       uint16_t W, const Conv &conv,
                       CBFPtr progress_cb = NULL);
```

---
## 11) Fully Connected (FCN)

All FCN variants compute:
- `y = W·x + b`
- then apply activation (`ACT_NONE` or `ACT_RELU` in typical usage; `ACT_SOFTMAX` exists but softmax is usually applied separately)

### 11.1 Memory (int8) → File (params from files)
```cpp
uint16_t noodle_fcn(const int8_t *input, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);
```

### 11.2 File (float text) → File (params from files)
```cpp
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    const char *out_fn, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);
```

### 11.3 Memory (float) → Memory (in-memory params)
```cpp
uint16_t noodle_fcn(const float *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNMem &fcn,
                    CBFPtr progress_cb = NULL);
```

### 11.4 Memory (byte) → Memory (params from files)
```cpp
uint16_t noodle_fcn(const byte *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);
```

### 11.5 Memory (int8) → Memory (params from files)
```cpp
uint16_t noodle_fcn(const int8_t *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);
```

### 11.6 File (float text) → Memory (params from files)
```cpp
uint16_t noodle_fcn(const char *in_fn, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb = NULL);
```

### 11.7 Memory (float) → Memory (params from files)
```cpp
uint16_t noodle_fcn(const float *input, uint16_t n_inputs, uint16_t n_outputs,
                    float *output, const FCNFile &fcn,
                    CBFPtr progress_cb);
```

---
## 12) Flatten

### 12.1 File → Memory flatten
```cpp
uint16_t noodle_flat(const char *in_fn, float *output, uint16_t V, uint16_t n_filters);
```
Reads `n_filters` feature maps from files named by `in_fn` template and writes a vector of length `V*V*n_filters`.

### 12.2 Memory → Memory flatten
```cpp
uint16_t noodle_flat(float *input, float *output, uint16_t V, uint16_t n_filters);
```

---
## 13) Activations

```cpp
uint16_t noodle_soft_max(float *input_output, uint16_t n);
uint16_t noodle_sigmoid(float *input_output, uint16_t n);
```

---
## 14) Misc utilities

### 14.1 Tensor slicing helper
```cpp
static inline float* noodle_slice(float* flat, size_t W, size_t z);
```
Slices a stacked `[Z, W, W]` tensor stored as contiguous planes.

### 14.2 Read first line of a text file
```cpp
void noodle_read_top_line(const char* fn, char *line, size_t maxlen);
```

---
## Notes 

- **Text file format:** Tensor/weight/bias I/O uses *human-readable text*, “one float per line”.
- **Filename templates are mutated in-place:** pass writable `char[]` buffers.
