 # <img src="./noodle.png" alt="Description" width="100"> Noodle Library Documentation

**Noodle** is a lightweight convolutional neural network (CNN) inference framework for embedded systems, optimized to operate with SD card storage and minimal RAM usage.

## API Reference

### Utility Functions

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



### Grid and Padding

- `void noodle_grid_to_file(byte *grid, char *fn, uint16_t n);`  
  Writes a square byte grid to a file.

- `float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P);`  
  Retrieves a value from a grid with padding.

### Core Computation

- `uint16_t noodle_do_bias(float *output, float bias, uint16_t n);`  
  Applies bias and ReLU activation to the output buffer.

- `uint16_t noodle_do_pooling(float *buffer, uint16_t W, uint16_t K, uint16_t S, char *fn);`  
  Applies 2×2 max pooling and writes output to file.

- `uint16_t noodle_do_convolution(byte *grid, float *kernel, uint16_t K, uint16_t W, float *output_buffer, uint16_t P, uint16_t S);`  
  Performs 2D convolution.

### File I/O

- `void noodle_read_from_file(char *fn, byte *buffer, uint16_t K, bool transposed = false);`  
  Reads byte matrix from file.

- `void noodle_read_from_file(char *fn, float *buffer, uint16_t K, bool transposed = false);`  
  Reads float matrix from file.

- `void noodle_reset_buffer(float *buffer, uint16_t n);`  
  Resets float buffer to zero.

### Composite Layer Functions

- `uint16_t noodle_conv(...)`  
  Full CNN pipeline (conv + bias + ReLU + pool) across multiple feature maps.

- `uint16_t noodle_flat(...)`  
  Flattens feature maps into a 1D buffer.

### Fully Connected (FCN) Layers

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

### Softmax

- `uint16_t noodle_soft_max(float *input_output, uint16_t n);`  
  Applies in-place softmax for classification output.

## License

MIT License



## Authors

- Auralius Manurung — Universitas Telkom, Bandung  
- Lisa Kristiana — ITENAS, Bandung
