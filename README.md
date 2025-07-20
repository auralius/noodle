<p align="center">
  <img src="./noodle.png" alt="Description" width="100"> 
</p>

<h1 align="center">Noodle 🍜≈🧠
  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16239228.svg)](https://doi.org/10.5281/zenodo.16239228)
  
</h1>


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
* With USPS dataset.
* Create and train the model, see [this Jupyter notebook.](https://deepnote.com/workspace/Noodle-7dbaf2d8-af0b-4fce-b6c2-3aa7fe49d539/project/NOODLE-93189612-1fd4-4b4d-beb6-ccb71c26052b/notebook/linear-svm-9f5e9548f30f4b65905aff8d546f3c06?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=93189612-1fd4-4b4d-beb6-ccb71c26052b)
* Copy the weights and biases to the SD Card.

__The Noodle implementation:__

```cpp
// Linear SVM is basically a 1-layer fully-connected network!
uint16_t V = noodle_fcn(GRID, 256, 10, OUTPUT_BUFFER, "svm-w.txt", "svm-b.txt", progress_hnd); //
```

---

### [Two layer fully connected network](#two-layer-fully-connected-network)
* With USPS dataset.
* Create and train the model, see [this Jupyter notebook.](https://deepnote.com/workspace/Noodle-7dbaf2d8-af0b-4fce-b6c2-3aa7fe49d539/project/NOODLE-93189612-1fd4-4b4d-beb6-ccb71c26052b/notebook/two-layer-fcn-b7010c875dae443ca39c0d68a9a7ea06?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=93189612-1fd4-4b4d-beb6-ccb71c26052b)
* Copy the weights and biases to the SD Card.

__The Noodle implementation:__

```cpp
// First fully connected (dense) layer:
// - Input: 256 input features from GRID (e.g., a flattened 16x16 image or feature map)
// - Output: 64 hidden units
// - Weights file: "fcn-w1.txt"
// - Bias file: "fcn-w2.txt"
// - Output is written to "out1.txt"
uint16_t V = noodle_fcn(GRID, 256, 64, "out1.txt", "fcn-w1.txt", "fcn-w2.txt", progress_hnd);

// Second (final) fully connected layer:
// - Input: 64 values from "out1.txt"
// - Output: 10 output units (for classification, e.g., digits 0–9)
// - Weights file: "fcn-w3.txt"
// - Bias file: "fcn-w4.txt"
// - Output is stored in OUTPUT_BUFFER (e.g., for softmax or decision)
// - V holds the final output size (should be 10)
V = noodle_fcn("out1.txt", V, 10, OUTPUT_BUFFER, "fcn-w3.txt", "fcn-w4.txt", progress_hnd);
```

---

### [LeNet-1](#lenet-1)
* With USPS dataset.
* Create and train the model, see [this Jupyter notebook.](https://deepnote.com/workspace/Noodle-7dbaf2d8-af0b-4fce-b6c2-3aa7fe49d539/project/NOODLE-93189612-1fd4-4b4d-beb6-ccb71c26052b/notebook/lenet-1-aaa1331bd09a4f578e7bb74ffa9576f3?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=93189612-1fd4-4b4d-beb6-ccb71c26052)
* Copy the weights and biases to the SD Card.

__The Noodle implementation:__

```cpp
// First convolution layer:
// - Input: 1 input channel (e.g., grayscale image), size 16x16
// - Output: 12 feature maps
// - Input file: "i1-x.txt"
// - Output file: "o1-x.txt"
// - Weights: "w1-x-x.txt", biases: "w2.txt"
// - Kernel size: 5x5, Padding: 2, Stride: 1
// - Max pooling: 2x2 with stride 2
// - Output dimension (V) is updated
V = noodle_conv(GRID, OUTPUT_BUFFER, 1, 12, "i1-x.txt", "o1-x.txt", "w1-x-x.txt", "w2.txt", 16, 2, 5, 1, 2, 2, progress_hnd);

// Second convolution layer:
// - Input: 12 feature maps from "o1-x.txt"
// - Output: 12 feature maps
// - Weights: "w3-x-x.txt", biases: "w4.txt"
// - Padding: 2 (possibly for preserving dimensions)
// - Max pooling: 2x2 with stride 2
// - Output stored in "o2-x.txt"
V = noodle_conv(GRID, OUTPUT_BUFFER, 12, 12, "o1-x.txt", "o2-x.txt", "w3-x-x.txt", "w4.txt", V, 2, 5, 1, 2, 2, progress_hnd);

// Flattening layer:
// - Converts the 12 pooled feature maps into a 1D vector
// - Output written into OUTPUT_BUFFER
// - V becomes total flattened vector length
V = noodle_flat("o2-x.txt", OUTPUT_BUFFER, V, 12);

// First fully connected layer:
// - Input size: V (from flattened feature maps)
// - Output: 30 neurons
// - Weights: "w5.txt", biases: "w6.txt"
// - ReLU activation enabled
// - Output saved in "o3.txt"
V = noodle_fcn(OUTPUT_BUFFER, V, 30, "o3.txt", "w5.txt", "w6.txt", true, progress_hnd);

// Output layer (fully connected):
// - Input: 30 values from "o3.txt"
// - Output: 10 class scores (for classification)
// - Weights: "w7.txt", biases: "w8.txt"
// - No activation (pure linear output)
// - Output stored in OUTPUT_BUFFER
V = noodle_fcn("o3.txt", V, 10, OUTPUT_BUFFER, "w7.txt", "w8.txt", false, progress_hnd);

// Softmax layer:
// - Applies softmax to the 10 output values for probabilistic interpretation
// - Operates in-place on OUTPUT_BUFFER
V = noodle_soft_max(OUTPUT_BUFFER, V);
```
---
  
### [LeNet-4](#lenet-4)  
* With MNIST dataset.
* Create and train the model, see [this Jupyter notebook.](https://deepnote.com/workspace/Noodle-7dbaf2d8-af0b-4fce-b6c2-3aa7fe49d539/project/NOODLE-93189612-1fd4-4b4d-beb6-ccb71c26052b/notebook/lenet-4-1202f3f64cc24a1dac3207cd35c7cf96?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=93189612-1fd4-4b4d-beb6-ccb71c26052b)
* Copy the weights and biases to the SD Card.

__The Noodle implementation:__

```cpp
// Perform the first convolutional layer:
// - Input: 1 channel (grayscale)
// - Output: 6 feature maps
// - Kernel: 5x5
// - Padding: 2 (to maintain spatial size)
// - Stride: 1
// - Pooling: 2x2, stride 2
// - Input file: "i1-x.txt"
// - Output file: "o1-x.txt"
// - Weights: "w1-x-x.txt", Bias: "w2.txt"
V = noodle_conv(GRID, OUTPUT_BUFFER, 1, 6, "i1-x.txt", "o1-x.txt", "w1-x-x.txt", "w2.txt",
                28, 2, 5, 1, 2, 2, progress_hnd);

// Perform the second convolutional layer:
// - Input: 6 feature maps (from previous layer)
// - Output: 16 feature maps
// - Kernel: 5x5
// - Padding: 0 (VALID convolution)
// - Stride: 1
// - Pooling: 2x2, stride 2
// - Input file: "o1-x.txt"
// - Output file: "o2-x.txt"
// - Weights: "w3-x-x.txt", Bias: "w4.txt"
V = noodle_conv(GRID, OUTPUT_BUFFER, 6, 16, "o1-x.txt", "o2-x.txt", "w3-x-x.txt", "w4.txt",
                V, 0, 5, 1, 2, 2, progress_hnd);

// Flatten the output feature maps from the last conv layer:
// - Reads from "o2-x.txt"
// - Combines all 16 feature maps into a single 1D array
// - Stores in OUTPUT_BUFFER
V = noodle_flat("o2-x.txt", OUTPUT_BUFFER, V, 16);

// First fully connected (dense) layer:
// - Input: flattened output from previous layer
// - Output: 120 hidden units
// - Weights: "w5.txt", Bias: "w6.txt"
// - Activation: ReLU
// - Output written to "o3.txt"
V = noodle_fcn(OUTPUT_BUFFER, V, 120, "o3.txt", "w5.txt", "w6.txt", progress_hnd);

// Second (final) fully connected layer:
// - Input: output from first FC layer ("o3.txt")
// - Output: 10 output units (for 10-class classification)
// - Weights: "w7.txt", Bias: "w8.txt"
// - Activation: ReLU
// - Output stored in OUTPUT_BUFFER
V = noodle_fcn("o3.txt", V, 10, OUTPUT_BUFFER, "w7.txt", "w8.txt", progress_hnd);
```

## [Authors](#authors)

- Auralius Manurung — Universitas Telkom, Bandung  
- Lisa Kristiana — ITENAS, Bandung
