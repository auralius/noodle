# Unified INT8 modification summary

- Float32 remains the default; existing source examples require no changes.
- `NOODLE_USE_INT8` selects int8 activations/weights and int32 bias/accumulation.
- `NoodleBuffer` and `NoodleTensor` use compile-time `NoodleData`.
- INT8 tensor metadata carries scale and zero point.
- Conv2D, Conv1D, Depthwise, ConvTranspose, FCN, pooling, GAP/GMP,
  flatten, concatenate, ReLU, sigmoid, and softmax have integer tensor paths.
- File, memory, near AVR PROGMEM Conv/Depthwise, and far AVR PROGMEM FCN
  parameter backends are represented.
- TFLite-style per-channel multiplier/shift requantization is included.
- The legacy q8-weight/float-activation option remains available only in
  Float32 mode.
- Legacy raw-pointer, file-to-file activation, and BatchNorm paths remain
  Float32-only; INT8 deployment uses the `NoodleTensor` API and folded BN.
