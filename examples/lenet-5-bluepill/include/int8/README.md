# Case Study 3: INT8 packed-FCN SD layout (Case 1 compatible)

This kit updates **only** the INT8 export layout and `src/main-int8.cpp`; it preserves the original FP32 build, raw 28×28 pixels, SRAM activations, Noodle INT8 arithmetic, input CRC, and stage timing.

## What “packed” means

- `w01.bin`–`w05.bin` remain **INT8 weight streams**, not embedded in one packed file.
- **Convolution 1 and 2** still use `wNN.bin`, `bNN.bin`, `mNN.bin`, `sNN.bin`. Latest Noodle convolution APIs use split metadata.
- **FC1–FC3** use `w03.bin` + `p03.bin`, `w04.bin` + `p04.bin`, `w05.bin` + `p05.bin`. A `pNN.bin` record is 3 little-endian signed 32-bit values: `(bias, multiplier, shift)` per output neuron.
- **14 .bin files, 64,302 bytes total**, compared with 20 files and the same number of bytes in the split case. This is lossless repacking; no quantized values change.

## Copy and flash

1. Copy **only** the 14 `.bin` files from `sd_int8_packed/` to a freshly cleared microSD root. Remove old `b03..05.bin`, `m03..05.bin`, `s03..05.bin`.
2. Replace the project's `src/main-int8.cpp` and `platformio.ini` with the supplied versions; keep `src/main-blue-pill.cpp` for FP32. If required, use the supplied `include/model_quant.h` from the validated INT8 exporter.
3. Use your unmodified **latest** `/home/auralius/works/noodle` source, *without applying the earlier FCN SRAM-cache patch*. It already implements packed `FCNFile.metadata_fn`.
4. Run `pio run -e bluepill_int8 -t upload` then reuse exactly the earlier sender command, with log `case3_int8_packed.log`.
5. Check `100/100` reference prediction agreement and monitor `STAGE fc1/fc2/fc3`; don't assume a speedup until measuring.

## Standard exporter integration

The updated `exporter/model_exporter.py` is a **minimal fork** of the uploaded current exporter. `--int8-sd-layout packed` now supports convolutional models by producing split conv metadata + packed FCN metadata. FP32 and `--int8-sd-layout split` retain their existing behavior.

```
python exporter/model_exporter.py lenet5_full_int8.tflite new_export_dir --int8-sd-layout packed
```

If you don't have the `.tflite` file locally, use `tools/pack_case3_int8.py` with your existing split deployment ZIP; it creates byte-identical packed FCN metadata without retraining or requantization.

## Validation and limitations

- All weights are byte-identical to the existing split export. All metadata triplets are verified on the host for exact equality.
- This has **not yet been compiled by PlatformIO or executed on Blue Pill** here. Hardware numerical/time equivalence must be checked using the same 100-image reference and SD card.
- No extra compile-time tuning flags are introduced. `NOODLE_USE_INT8` and existing hardware/semantic flags remain.

## Pure TensorFlow notebook

`Case3_LeNet5_Pure_TensorFlow_FP32_INT8_PackedFCN.ipynb` preserves the original pure-TF FP32+INT8 pipeline and existing_sd/train_new modes. Its Git exporter split output remains only an intermediate, and the final deployment ZIP contains 14 mixed-layout files plus a corrected manifest. No retraining is needed for the already verified first-run parameter ZIP.
