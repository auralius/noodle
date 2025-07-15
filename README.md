 # <img src="./noodle.png" alt="Description" width="100"> noodle

Lightweight file-based Convolutional Neural Network (CNN) implementation for embedded systems.
_noodle_ relies on SD Card and memory-reuse for efficient implementation.

- **Function List**

  - `float noodle_read_float(File &f)`
    - Reads a float value from a file.

  - `void noodle_delete_file(char *fn)`
    - Deletes a file from the SD card.

  - `float* noodle_create_buffer(uint16_t size)`
    - Allocates a buffer for float data.

  - `void noodle_delete_buffer(float *buffer)`
    - Frees a previously allocated float buffer.

  - `void noodle_grid_to_file(byte *grid, char *fn, uint16_t n)`
    - Writes a grid of byte data to a file.

  - `float noodle_get_padded_x(byte *grid, int16_t i, int16_t j, int16_t W, int16_t P)`
    - Gets a padded value from a byte grid.

  - `uint16_t noodle_do_bias(float *output, float bias, uint16_t n)`
    - Applies bias and ReLU activation to an output buffer.

  - `uint16_t noodle_do_pooling(float *buffer, uint16_t W, uint16_t K, uint16_t S, char *fn)`
    - Performs max pooling and stores result in a file.

  - `uint16_t noodle_do_convolution(byte *grid, float *kernel, uint16_t K, uint16_t W, float *output_buffer, uint16_t P, uint16_t S)`
    - Performs 2D convolution on a byte grid.

  - `void noodle_read_from_file(char *fn, byte *buffer, uint16_t K, bool transposed = false)`
    - Reads a byte matrix from a file.

  - `void noodle_read_from_file(char *fn, float *buffer, uint16_t K, bool transposed = false)`
    - Reads a float matrix from a file.

  - `void noodle_reset_buffer(float *buffer, uint16_t n)`
    - Resets a float buffer to zero.

  - `uint16_t noodle_conv(...)`
    - Executes full convolution, bias, ReLU, and pooling pipeline.

  - `uint16_t noodle_flat(char *in_fn, float *output_buffer, uint16_t V, uint16_t n_filters)`
    - Flattens feature maps into a 1D vector.

  - `uint16_t noodle_fcn(...)`
    - Fully connected layer operation with multiple overloads:
      - From buffer to file (float input)
      - From buffer to file (byte input)
      - From buffer to buffer (byte input)
      - From buffer to buffer (float input)
      - From file to buffer

- **Data Types**

  - `CBFPtr`: Function pointer type for progress callbacks.

- **External Variables**

  - `SdFat SD`: Global SD card object.

- **License**

  - MIT License
  - © Auralius Manurung (Universitas Telkom)
  - © Lisa Kristiana (ITENAS)


