# Changelog

## v1.0.2

### Features

- Optimized IRAM usage on ESP32-S3

## v1.0.1

### Features

- Supported IDF6.0
- Supported ESP32-S31, ESP32-H4, ESP32-C61

### Bug Fixes

- Fixed build error in test_app

## v1.0.0

### Features

- Added support for RGB565(big endian), RGB565(little endian), and CbYCrY pixel format in the encoder
- Aligned pixel format with esp-fourcc

### Bug Fixes

- Fixed encoding error with large gray images
- Fixed missing assignment of the `out_size` member in the `jpeg_dec_io_t` structure when scale is enabled during decoding

## v0.6.1

### Features

- Optimize error handling logic of the decoder
- Improve example of encoder block mode in test_app

### Bug Fixes

- Fix decoding error for some images containing restart markers

## v0.6.0

### Features

- Support IDF5.4
- Support SD power supply comes from internal LDO IO in test_app
- Add 16-byte alignment check for encoder input buffer when using ESP32-S3
- Add 16-byte alignment check for decoder output buffer when using ESP32-S3

### Bug Fixes

- Fix inaccurate `inbuf_remain` member in `jpeg_dec_io_t` structure after decoding is finished
- Fix spelling errors in test_app

## v0.5.1

### Bug Fixes

- Fix memory leakage on jpeg_enc_open fail
- Fix memory allocation fail when dram is insufficient

## v0.5.0

### Features

- Initial version of `esp_new_jpeg`
- Add JPEG encoder and decoder library
