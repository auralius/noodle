 # <img src="./noodle.png" alt="Description" width="100"> noodle


## 📘 API Reference

Lightweight CNN-style operations using SD card file storage on embedded systems (e.g., Arduino).
_noodle_ relies on SD Card and memory-reuse for efficient implementation.

---

### 🧮 Core Functions

#### `float noodle_read_float(File &f)`
Reads a floating-point value from a file.

---

#### `void noodle_delete_file(char *fn)`
Deletes a file from the SD card.  
**@param** `fn` Filename to delete.

---

#### `float *noodle_create_buffer(uint16_t size)`
Allocates memory for a float buffer.  
**@param** `size` Number of floats.  
**@return** Pointer to the allocated buffer.

---

#### `void noodle_delete_buffer(float *buffer)`
Frees the buffer memory.  
**@param** `buffer` Pointer to a previously allocated buffer.

---

#### `void noodle_grid_to_file(byte *grid, char *fn, uint16_t n)`
Stores a grid (n x n) of bytes into a file.  
**@param** `grid` Pointer to input grid.  
**@param** `fn` Output filename.  
**@param** `n` Grid dimension.

---

#### `float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P)`
Reads a padded element from the grid.  
**@param** `grid` Input grid.  
**@param** `i, j` Row and column indices (with padding).  
**@param** `W` Width of original grid.  
**@param** `P` Padding amount.  
**@return** Padded value or 0.0 if out of bounds.

---

#### `uint16_t noodle_do_bias(float *output, float bias, uint16_t n)`
Applies bias and ReLU activation on an output grid (n x n).  
**@param** `output` Pointer to output grid.  
**@param** `bias` Scalar bias value.  
**@param** `n` Grid dimension.  
**@return** Output dimension (unchanged).

---

#### `uint16_t noodle_do_pooling(...)`
Performs max pooling and writes to file.  
**@param** `output` Pointer to input float grid.  
**@param** `W` Width of input.  
**@param** `K` Pooling kernel size.  
**@param** `S` Pooling stride.  
**@param** `fn` Output filename.  
**@return** Output dimension after pooling.

---

#### `uint16_t noodle_do_convolution(...)`
Performs convolution with padding and stride.  
**@param** `grid` Input byte grid.  
**@param** `kernel` Convolution kernel.  
**@param** `K` Kernel size.  
**@param** `W` Input width.  
**@param** `output_buffer` Output buffer (float).  
**@param** `P` Padding.  
**@param** `S` Stride.  
**@return** Output dimension.

---

### 📂 File I/O

#### `void noodle_read_from_file(char *fn, byte *buffer, uint16_t K, bool transposed = false)`
Reads a K x K byte matrix from file.  
**@param** `fn` Filename.  
**@param** `buffer` Output buffer.  
**@param** `K` Matrix size.  
**@param** `transposed` If true, transpose matrix while reading.

---

#### `void noodle_read_from_file(char *fn, float *buffer, uint16_t K, bool transposed = false)`
Same as above, for float matrices.

---

#### `void noodle_reset_buffer(float *buffer, uint16_t n)`
Fills a float buffer with zeros.  
**@param** `buffer` Float buffer.  
**@param** `n` Number of elements.

---

### 🧠 CNN Pipeline

#### `uint16_t noodle_conv(...)`
Performs convolution + bias + pooling for multiple input/filter combinations.  
**@param** `grid` Input byte grid.  
**@param** `output_buffer` Reusable float buffer.  
**@param** `n_inputs` Number of input feature maps.  
**@param** `n_filters` Number of filters.  
**@param** `in_fn`, `out_fn`, `weight_fn`, `bias_fn` Filenames for I/O.  
**@param** `W`, `P`, `K`, `S`, `M`, `T` Sizes for input, padding, kernel, stride, pooling kernel, pooling stride.  
**@param** `progress_cb` Optional progress callback.  
**@return** Output dimension after pooling.

---

#### `uint16_t noodle_flat(float *output_buffer, char *in_fn, uint16_t V, uint16_t n_filters)`
Flattens `n_filters` V×V matrices into a 1D vector.  
**@param** `output_buffer` Output vector.  
**@param** `in_fn` Input filename pattern.  
**@param** `V` Dimension of feature map.  
**@param** `n_filters` Number of filters.  
**@return** Length of flattened vector.

---

#### `uint16_t noodle_fcn(...)`
Fully connected layer variant 1 (float input).  
**@param** `output_buffer` Input/output buffer.  
**@param** `n_inputs` Number of inputs.  
**@param** `n_outputs` Number of outputs.  
**@param** `out_fn` Output filename.  
**@param** `weight_fn` Weight file.  
**@param** `bias_fn` Bias file.  
**@param** `progress_cb` Optional progress callback.

---

#### `uint16_t noodle_fcn(byte *output_buffer, ...)`
FCN variant 2 (byte input buffer).

---

#### `uint16_t noodle_fcn(char *in_fn, ..., float *output_buffer, ...)`
FCN variant 3: reads input vector from file, writes result to buffer.

---
