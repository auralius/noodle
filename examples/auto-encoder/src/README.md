# Deep Sequential AE-28x28 source layout

This source folder holds two versions of the same 28x28 denoising
auto-encoder experiment.

The main story is simple: a host sends one noisy MNIST-sized grayscale image to
the ESP32-S3, the board reconstructs a cleaner image, and the host records the
time, output pixels, and memory state. `main.cpp` runs the topology directly
with Noodle. `main_tflm_denoising_benchmark.cpp` runs the same job through
TFLite Micro so we can compare Noodle against a widely used embedded inference
runtime under the same serial protocol.

## Source files

- `main.cpp`
  - Noodle implementation of the Deep Sequential-AE-28x28 forward pass.
  - Uses generated `w01.h` ... `w11.h` and `b01.h` ... `b11.h` parameter
    headers from `include/`.
  - Keeps two grow-only `NoodleTensor` buffers, `A` and `B`, and swaps them
    between layers to avoid repeated allocation during inference.

- `main_tflm_denoising_benchmark.cpp`
  - TFLite Micro benchmark firmware for the same denoising task.
  - Uses `include/model_data.h`, which must provide `g_model[]` and
    `g_model_len`.
  - Allocates a fixed tensor arena and reports memory before arena allocation,
    after arena allocation, after `AllocateTensors()`, and during later `MEM`
    queries.

- `noodle_serial.cpp`
  - Shared serial protocol helper used by both firmware variants.
  - Keeps the benchmark harness identical: the host sends `IMG`, receives
    `OUT <microseconds>`, reads exactly 784 output bytes, then waits for
    `READY`.

## Auto-encoder topology

The model is a symmetric convolutional denoising auto-encoder for a single
28x28 grayscale image. The encoder compresses the image through two spatial
downsampling steps, and the decoder expands it back to the original image size.

| Layer | Operation | Output shape |
| --- | --- | --- |
| input | uint8 image normalized to float `[0, 1]` | `28 x 28 x 1` |
| L01 | `Conv2D`, 3x3, stride 1, 16 channels, ReLU | `28 x 28 x 16` |
| L02 | `Conv2D`, 3x3, stride 1, 16 channels, ReLU | `28 x 28 x 16` |
| L03 | `Conv2D`, 3x3, stride 2, 32 channels, ReLU | `14 x 14 x 32` |
| L04 | `Conv2D`, 3x3, stride 1, 32 channels, ReLU | `14 x 14 x 32` |
| L05 | `Conv2D`, 3x3, stride 2, 64 channels, ReLU | `7 x 7 x 64` |
| L06 | `Conv2D`, 3x3, stride 1, 64 channels, ReLU | `7 x 7 x 64` |
| L07 | `Conv2DTranspose`, 3x3, stride 2, 32 channels, ReLU | `14 x 14 x 32` |
| L08 | `Conv2D`, 3x3, stride 1, 32 channels, ReLU | `14 x 14 x 32` |
| L09 | `Conv2DTranspose`, 3x3, stride 2, 16 channels, ReLU | `28 x 28 x 16` |
| L10 | `Conv2D`, 3x3, stride 1, 16 channels, ReLU | `28 x 28 x 16` |
| L11 | `Conv2D`, 3x3, stride 1, 1 channel | `28 x 28 x 1` |
| output | sigmoid, clamp to `[0, 1]`, quantize to uint8 | `784 bytes` |

The largest visible activation in the Noodle path is `28 x 28 x 16` floats.
With the two ping-pong tensors preallocated, the expected `A + B` tensor
capacity is `2 * 12544` floats, or `100352` bytes.

## Benchmark intent

TFLite Micro is here as the reference point, not as a separate demo. Both
firmwares should answer the same question:

```text
For the same denoising topology, input images, board, and serial harness,
what does each runtime cost in latency, memory, and output quality?
```

The comparison is strongest when these stay fixed:

- the same trained model/export is used for the Noodle headers and TFLite
  flatbuffer;
- the same ESP32-S3 board, flash/PSRAM settings, and partition file are used;
- the same host script, dataset slice, noise settings, and image order are used;
- the host records `OUT <microseconds>`, output MSE, and `MEM` fields for both
  firmware builds.

## PlatformIO environments

`platformio.ini` selects which firmware source is compiled.

```bash
pio run -e esp32-s3-n16r8
pio run -e esp32-s3-n16r8 -t upload
```

builds and uploads the Noodle path from `main.cpp`.

```bash
pio run -e esp32-s3-n16r8-tflm
pio run -e esp32-s3-n16r8-tflm -t upload
```

builds and uploads the TFLite Micro path from
`main_tflm_denoising_benchmark.cpp`.

## Host-side run

The host driver lives in `test/sender.py`. It sends noisy 28x28 images in
64-byte chunks, waits for the reconstructed 784-byte output image, computes MSE
against the clean image, and optionally asks the board for memory with `MEM`.

Use the same sender settings for both firmware variants when collecting a
benchmark pair. If both runs write CSV logs, change `LOG_CSV_PATH` between runs
so the Noodle and TFLite Micro measurements do not overwrite each other.
