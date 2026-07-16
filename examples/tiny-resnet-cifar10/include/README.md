# RP2350 Tiny FireNet CIFAR-10 serial demo

Files:

- `main_rp2350_firenet.cpp`  
  Firmware for the RP2350/Pico-style board using NoodleBuffer.

- `cifar10_rgb_sender.py`  
  Python sender using the same `NoodleSerial` protocol. It sends 32×32 RGB CIFAR-10 images.

- `noodle_serial.h`, `noodle_serial.cpp`  
  Same chunked serial helper style as the previous Blue Pill demo.

## Required model files

Copy the exported headers from the Colab notebook into the firmware project:

```text
model_weights.h
w01.h ... w11.h
b01.h ... b11.h
```

The firmware includes:

```cpp
#include "model_weights.h"
```

## Important build flags

Use max pooling and no filesystem backend:

```ini
build_flags =
  -DNOODLE_USE_NONE
  -DNOODLE_POOL_MODE=NOODLE_POOL_MAX
```

## Serial protocol

Payload size is now RGB:

```text
32 * 32 * 3 = 3072 bytes
```

The host sends RGB bytes in HWC order:

```text
[y][x][r,g,b]
```

The firmware converts it to Noodle CHW float:

```text
[c][y][x], normalized to [0, 1]
```

## Sender example

```bash
python cifar10_rgb_sender.py --port /dev/ttyACM0 --baud 921600 --n 20 --random
```

The sender downloads CIFAR-10 to `./cifar10_local` when the local cache is missing.
It also saves the sent images as PNG files in `./sent_images`.


## v2 note

This version uses the public Noodle functions already available in the latest Noodle zip:

```cpp
noodle_concat(&E1, C_e1, &E3, C_e3, output, W);
noodle_gap(&B, C, W);
```

The only local helper still kept in the firmware is `maxpool2d_chw()`, because this is ordinary spatial MaxPool2D after Fire2. It is different from `noodle_gmp()`, which is global max pooling.
