# Noodle — Case Study 3, INT8 FCN SD-access optimization (Blue Pill)

This is a **one-file patch** against the user-supplied `noodle-latest.zip` (13 October 2026). No changes to the FP32 firmware, the Blue Pill `main-int8.cpp`, the TFLite model, the SD binaries, the serial sender, or PlatformIO flags are necessary.

## What it fixes

The original split-layout `FCNFile` code in `src/noodle_int8_layers.cpp` opened/searched/closed `bNN.bin`, `mNN.bin`, and `sNN.bin` **once per output neuron**. The Blue Pill LeNet-5 FC layers therefore needed 360 + 252 + 30 = **642 metadata opens per image**. The weight file was already streamed sequentially and remains unchanged.

On **non-AVR systems with BIN parameter files and SRAM-resident input/output**, this patch:

1. Allocates a temporary buffer of **12 bytes per output neuron** (three `int32` metadata arrays).
2. Reads each of the three metadata files sequentially once (only **three metadata opens per FCN layer**, **nine per inference**).
3. Closes the metadata handle before streaming the INT8 weight file, retaining the original two-handle maximum.
4. Reuses the exact existing INT8 MAC, bias addition, multiplier/shift, activation limits, and output quantization logic.
5. Frees temporary metadata storage when the FCN function returns.

If allocation fails, it **automatically falls back** to the original per-neuron metadata path. AVR targets retain the old low-SRAM path. Packed metadata and text formats are unchanged. No additional optimization flag or public API is introduced.

| FC layer | Original metadata opens | Optimized when allocation succeeds | Temporary cache |
|---|---:|---:|---:|
| FC1 (400→120) | 360 | 3 | 1,440 B |
| FC2 (120→84) | 252 | 3 | 1,008 B |
| FC3 (84→10) | 30 | 3 | 120 B |
| **Total** | **642** | **9** | one layer at a time |

## Apply

Back up your current library source first:

```bash
cp /home/auralius/works/noodle/src/noodle_int8_layers.cpp \
   /home/auralius/works/noodle/src/noodle_int8_layers.cpp.before_case3_cache
```

Copy `src/noodle_int8_layers.cpp` from this kit to the same path, **or** apply the provided unified diff from your library root:

```bash
cd /home/auralius/works/noodle
patch -p1 --dry-run < /path/to/int8-fcn-metadata-cache.patch
patch -p1 < /path/to/int8-fcn-metadata-cache.patch
```

**Do not both copy and apply the patch.** The diff is against the attached `noodle-latest.zip` specifically; a newer local commit may require reconciliation.

## Build and verify: same Case Study 3 conditions

Keep `src/main-int8.cpp`, `src/main-blue-pill.cpp`, and `platformio.ini` untouched. Do **not** regenerate parameters, change the SD interface/clock, or alter the 100 MNIST inputs.

From the Case Study 3 PlatformIO project:

```bash
pio run -e bluepill_int8 -t upload
python test/blue-pill-sender.py \
  --csv test/mnist_train.csv \
  --port /dev/ttyUSB0 \
  --mode int8 \
  --count 100 \
  --reference tools/case3_lenet5_fp32_vs_int8_first100.csv \
  --log case3_int8_fcn_cache.log
python tools/compare_int8_serial.py \
  case3_int8_fcn_cache.log \
  tools/case3_lenet5_fp32_vs_int8_first100.csv
```

**Acceptance criteria**: 100/100 input CRC, 100/100 predictions matched to the original INT8 TFLite reference, 99/100 correct on the first 100 training samples, and the same or better output quantization agreement as before (baseline 96/100 exact softmax vectors, maximum 1 INT8 step). Record timings for `conv1_pool1`, `conv2_pool2`, `flatten`, `fc1`, `fc2`, `fc3`, and `softmax`; use the same board, SD card, and clock.

Before optimization, the existing hardware log reported `fc1=2.328875 s`, `fc2=1.531744 s`, `fc3=0.183716 s` and total `4.913 s`. **This patch has not been measured on the Blue Pill yet.** The actual runtime speedup and available heap must be confirmed by the user.

## Offline validation completed

- The patched `noodle_int8_layers.cpp` passed C++17 syntax checking in Noodle's NONE-backend build with full INT8 enabled.
- The patch modifies only SD metadata access and dispatch; all multiply/accumulate/requantization formulas are unchanged.
- `tools/check_metadata.py` checked all 214 FC-layer outputs across `b03..05`, `m03..05`, `s03..05` against the original per-neuron reads using the real supplied INT8 binaries.
- There is **no claim of full PlatformIO compilation or hardware timing verification** in this kit.

To reproduce the metadata check from this kit's root, provide a directory holding the original 20 INT8 SD files:

```bash
python tools/check_metadata.py /path/to/sd_int8
```
