# RP2350 TinyResNet CIFAR-10 with Noodle

## Files

- `src/main.cpp`: TinyResNet residual inference firmware.
- `src/noodle_serial.h/.cpp`: existing IMG/RDYIMG/ACK serial protocol.
- `lib/noodle/`: the supplied Noodle library.
- `tinyresnet_cifar10_rp2350_export.ipynb`: patched Colab notebook.

## Training and export

1. Open `tinyresnet_cifar10_rp2350_export.ipynb` in Colab.
2. Select a GPU runtime.
3. Run all cells. The default is 20 epochs.
4. Download `tinyresnet_export.zip`.
5. Copy the generated
   `tinyresnet_cifar10_export/model_weights_tinyresnet.h`
   into this project's `src/` directory.

The patched exporter adds:

```cpp
static const float input_mean[3];
static const float input_inv_std[3];
```

These values reproduce the Keras `Normalization` layer on the RP2350.

## Residual memory schedule

Five NoodleBuffer objects are used:

- `X`: normalized input;
- `A`, `B`: alternating residual outputs;
- `C`: second-convolution output;
- `D`: projected shortcut for stride-2 blocks.

The buffers are preallocated before inference so that arena relocation cannot
invalidate a live shortcut during a residual block.

Expected retained activation arena:

- `X`: 3,072 floats;
- `A`: 16,384 floats;
- `B`: 16,384 floats;
- `C`: 16,384 floats;
- `D`: 8,192 floats.

Total: 60,416 floats = 241,664 bytes.

## Required compile definitions

```text
-DNOODLE_USE_NONE
-DNOODLE_POOL_MODE=NOODLE_POOL_NONE
```

`ACT_NONE` is used for the second convolution and the projection shortcut;
ReLU is applied only after residual addition.

## Serial protocol

The protocol is unchanged from the FireNet firmware:

```text
IMG
RDYIMG
<3072 RGB bytes, 64 bytes/chunk>
ACK
...
PRED <class_id> <seconds> <confidence> <class_name>
READY
```
