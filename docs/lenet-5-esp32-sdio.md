## SD-Card–Backed LeNet-5 Inference

This section evaluates a **LeNet-5–style inference pipeline** in which **weights, biases, and intermediate activations are streamed to and from an SD card**, rather than retained fully in RAM. The target platform is an **ESP32-P4-Pico**, interfacing with external storage via **1-bit SDIO**.

| Step | Op / Layer                        | Input (where)            | Output (where)           | Parameters (where)               | Notes                                                                  |
| ---: | --------------------------------- | ------------------------ | ------------------------ | -------------------------------- | ---------------------------------------------------------------------- |
|    1 | Conv1 + Pool                      | `GRID` (**variable**)    | `out1.txt` (**file**)    | `w01.txt`, `b01.txt` (**files**) | `noodle_conv_float("in1.txt", 1, 6, "out1.txt", 28, cnn1, pool, ...)`  |
|    2 | Conv2 + Pool                      | `out1.txt` (**file**)    | `out2.txt` (**file**)    | `w02.txt`, `b02.txt` (**files**) | `noodle_conv_float("out1.txt", 6, 16, "out2.txt", V, cnn2, pool, ...)` |
|    3 | Flatten (CHW file → HWC-flat RAM) | `out2.txt` (**file**)    | `BUFFER1` (**variable**) | —                                | `V = noodle_flat("out2.txt", BUFFER1, V, 16)`                          |
|    4 | FC1 (RAM → file)                  | `BUFFER1` (**variable**) | `out3.txt` (**file**)    | `w03.txt`, `b03.txt` (**files**) | `V = noodle_fcn(BUFFER1, V, 120, "out3.txt", fcn1, ...)`               |
|    5 | FC2 (file → file)                 | `out3.txt` (**file**)    | `out4.txt` (**file**)    | `w04.txt`, `b04.txt` (**files**) | `V = noodle_fcn("out3.txt", V, 84, "out4.txt", fcn2, ...)`             |
|    6 | FC3 + Softmax (file → RAM)        | `out4.txt` (**file**)    | `BUFFER2` (**variable**) | `w05.txt`, `b05.txt` (**files**) | `V = noodle_fcn("out4.txt", V, 10, BUFFER2, fcn3, ...)`                |

![](attachments/p4-sdcard.png)