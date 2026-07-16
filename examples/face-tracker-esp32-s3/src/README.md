# TinyFaceCenter Flatten ESP32-S3 + Noodle


```text
conv4 -> noodle_flat -> Dense64 -> Output4
```

The model output is:

```text
[p, x01, y01, s]
```

where `x01,y01` are absolute normalized face-center coordinates in the 96x96 camera frame.

For the OLED marker:

```cpp
cx = roundf(x01 * 63.0f);
cy = 32 + roundf(y01 * 95.0f);
```

For motor control later:

```cpp
dx = x01 - 0.5f;
dy = y01 - 0.5f;
```

## PlatformIO

Copy `main-camera-tinyfacecenter-flatten.cpp` into `src/`, or rename it to match your
`build_src_filter`.

Copy `tinyfacecenter_flatten_weights.h` into the same include path as your main source, or into
your project's `include/` folder.
