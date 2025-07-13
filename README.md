 # <img src="./noodle.png" alt="Description" width="100"> noodle

_noodle_ is an implementation of the forward pass of CNN layers for Arduino. 
_noodle_ relies on SD Card and memory-reuse for efficient implementation.

 ## How to use
 ### 
 ```cpp
 float noodle_read_float(File &f)
 ```

 ### 
 ```cpp
 noodle_delete_file(char *fn)
 ```
 
 ### 
 ```cpp
 float *noodle_create_buffer(uint16_t size)
 ```

 ### 
 ```cpp

void noodle_delete_buffer(float *buffer) 
```

### 
```cpp
void noodle_grid_to_file(byte *grid, char *fn,
                         uint16_t n)
```
### 
```cpp
void noodle_grid_to_file(byte *grid,
                         char *fn, uint16_t n) 
```

### 
```cpp
float noodle_get_padded_x(byte *grid,
                          int16_t i,
                          int16_t j,
                          int16_t W,
                          int16_t P) 
```

### 
```cpp
uint16_t noodle_do_bias(float *output, float bias, uint16_t n) 
```

### 
```cpp
uint16_t noodle_do_pooling(float *output,
                           uint16_t W,
                           uint16_t K,
                           uint16_t S,
                           char *fn) 
```

##
* Input size is `W x W`.
* The kernel filter size is `K x K`.
* The padding is `P` (uniform and zero padding).
* The stride length is `S`.
```cpp
uint16_t noodle_do_convolution(byte *grid,
                               float *kernel,
                               uint16_t K,
                               uint16_t W,
                               float *output_buffer,
                               uint16_t P,
                               uint16_t S) 
```

##
Load a BYTE  square matrix from a file `(K x K)`. The matrix was previously stored linearly
```cpp
void noodle_read_from_file(char *fn,
                           byte *buffer,
                           uint16_t K,
                           bool transposed = false) 
```

##
```cpp
void noodle_reset_buffer(float *buffer,
                         uint16_t n) {
```

##
```cpp
uint16_t noodle_conv(byte *grid,
                     float *output_buffer,
                     uint16_t n_inputs,
                     uint16_t n_filters,
                     char *in_fn, char *out_fn,
                     char *weight_fn, char *bias_fn,
                     uint16_t W,  // Input size (W x W)
                     uint16_t P,  // Number of uniform-zero-padding layer for the input
                     uint16_t K,  // Convolution kernel size (K x K)
                     uint16_t S,  // Convolution stride length
                     uint16_t M,  // Max-pooling filter size
                     uint16_t T,  // Max-poooling stride length
                     CBFPtr progress_cb=NULL)
```


##
Flattening, from a several input files to output_buffer
```cpp
uint16_t noodle_flat(float *output_buffer, 
                     char *in_fn, 
                     uint16_t V, 
                     uint16_t n_filters) 
```

##
From RAM to SD Card
```cpp
uint16_t noodle_fcn(float *output_buffer,
                    uint16_t n_inputs,
                    uint16_t n_outputs,
                    char *out_fn,
                    char *weight_fn,
                    char *bias_fn,
                    CBFPtr progress_cb=NULL) 
```

##
From SD Card to RAM
```cpp
uint16_t noodle_fcn(char *in_fn, 
                    uint16_t n_inputs, 
                    uint16_t n_outputs, 
                    float *output_buffer, 
                    char *weight_fn, 
                    char *bias_fn,
                    CBFPtr progress_cb=NULL) 
```
