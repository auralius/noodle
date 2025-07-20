<p align="center">
  <img src="./noodle.png" alt="Description" width="100"> 
</p>

<h1 align="center">Noodle 🍜≈🧠</h1>


**Noodle** is a lightweight convolutional neural network (CNN) inference framework for embedded systems, optimized to operate with SD card storage and minimal RAM usage.

## [Table of Contents](#table-of-contents)
* [API Reference](#api-reference)
  * [Utility Functions](#utility-functions)
  * [Grid and Padding](#grid-and-padding)
  * [Core Computation](#core-computation)
  * [File I/O](#file-io)
  * [Composite Layer Functions](#composite-layer-functions)
  * [Fully Connected Layers](#fully-connected-layers)
  * [Softmax](#softmax)
* [Examples](#examples)
  * [Linear SVM](#linear-svm)
  * [Two layer fully connected network](#two-layer-fully-connected-network)
  * [LeNet-1](#lenet-1)
  * [LeNet-4](#lenet-4)  
* [License](#license)
* [Authors](#authors)

## [API Reference](#api-reference)

### [Utility Functions](#utility-functions)

- `void noodle_write_float(File &f, float d);`  
  Writes a float to a file.

- `float noodle_read_float(File &f);`  
  Reads a float value from a file.

- `void noodle_delete_file(char *fn);`  
  Deletes a file from the SD card.

- `float* noodle_create_buffer(uint16_t size);`  
  Allocates a float buffer.

- `void noodle_delete_buffer(float *buffer);`  
  Frees a float buffer.



### [Grid and Padding](#grid-and-padding)

- `void noodle_grid_to_file(byte *grid, char *fn, uint16_t n);`  
  Writes a square byte grid to a file.

- `float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P);`  
  Retrieves a value from a grid with padding.

### [Core Computation](#core-computation)

- `uint16_t noodle_do_bias(float *output, float bias, uint16_t n);`  
  Applies bias and ReLU activation to the output buffer.

- `uint16_t noodle_do_pooling(float *buffer, uint16_t W, uint16_t K, uint16_t S, char *fn);`  
  Applies 2×2 max pooling and writes output to file.

- `uint16_t noodle_do_convolution(byte *grid, float *kernel, uint16_t K, uint16_t W, float *output_buffer, uint16_t P, uint16_t S);`  
  Performs 2D convolution.

### [File I/O](#file-io)

- `void noodle_read_from_file(char *fn, byte *buffer, uint16_t K, bool transposed = false);`  
  Reads byte matrix from file.

- `void noodle_read_from_file(char *fn, float *buffer, uint16_t K, bool transposed = false);`  
  Reads float matrix from file.

- `void noodle_reset_buffer(float *buffer, uint16_t n);`  
  Resets float buffer to zero.

### [Composite Layer Functions](#composite-layer-functions)

- `uint16_t noodle_conv(...)`  
  Full CNN pipeline (conv + bias + ReLU + pool) across multiple feature maps.

- `uint16_t noodle_flat(...)`  
  Flattens feature maps into a 1D buffer.

### [Fully Connected Layers](#fully-connected-layers)

Supports multiple FCN overloads depending on input/output type:

```cpp
uint16_t noodle_fcn(float *input_buffer, uint16_t n_inputs, uint16_t n_outputs, char *out_fn, ...);
uint16_t noodle_fcn(byte *input_buffer, uint16_t n_inputs, uint16_t n_outputs, char *out_fn, ...);
uint16_t noodle_fcn(byte *input_buffer, ..., float *output_buffer, ...);
uint16_t noodle_fcn(float *input_buffer, ..., float *output_buffer, ...);
uint16_t noodle_fcn(char *in_fn, ..., float *output_buffer, ...);
```
| Variant           | Input    | Output  | 
| ----------------- | -------- | ------- | 
| `float* → file`   | RAM      | SD file | 
| `byte* → file`    | RAM      | SD file | 
| `byte* → buffer`  | RAM      | RAM     | 
| `float* → buffer` | RAM      | RAM     | 
| `file → buffer`   | SD File  | RAM     | 



All support optional ReLU and progress callback.

### [Softmax](#softmax)

- `uint16_t noodle_soft_max(float *input_output, uint16_t n);`  
  Applies in-place softmax for classification output.

## [Examples](#examples)

### [Linear SVM](#linear-svm)

### [Two layer fully connected network](#two-layer-fully-connected-network)

### [LeNet-1](#lenet-1)

### [LeNet-4](#lenet-4)  

## [License](#license)

[MIT License](https://mit-license.org/)

## [Authors](#authors)

- Auralius Manurung — Universitas Telkom, Bandung  
- Lisa Kristiana — ITENAS, Bandung
