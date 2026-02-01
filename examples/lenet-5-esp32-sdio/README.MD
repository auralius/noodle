This section is dedicated to benchmark layer functions that offload weights, biases, and tensors to an SD card. The system utilizes an ESP32-P4-Pico interfacing with the storage via 4-bit SDIO.

| Step | Op / Layer                                 | Input (where)         | Output (where)        | Parameters (where)               | Notes                                                                   |
| ---: | ------------------------------------------ | --------------------- | --------------------- | -------------------------------- | ------------------------------------------------------------------------|
|    0 | `noodle_grid_to_file(GRID, "in1.txt", 28)` | `GRID` (**var**)      | `in1.txt` (**file**)  | —                                | Received image arrives in RAM transfers to SD                           |
|    1 | Conv1 + Pool                               | `in1.txt` (**file**)  | `out1.txt` (**file**) | `w01.txt`, `b01.txt` (**files**) | `noodle_conv_float("in1.txt", 1, 6, "out1.txt", 28, cnn1, pool, ...)`   |
|    2 | Conv2 + Pool                               | `out1.txt` (**file**) | `out2.txt` (**file**) | `w02.txt`, `b02.txt` (**files**) | `noodle_conv_float("out1.txt", 6, 16, "out2.txt", V, cnn2, pool, ...)`  |
|    3 | Flatten (CHW file → HWC-flat RAM)          | `out2.txt` (**file**) | `BUFFER1` (**var**)   | —                                | `V = noodle_flat("out2.txt", BUFFER1, V, 16)`                           |
|    4 | FC1 (RAM → file)                           | `BUFFER1` (**var**)   | `out3.txt` (**file**) | `w03.txt`, `b03.txt` (**files**) | `V = noodle_fcn(BUFFER1, V, 120, "out3.txt", fcn1, ...)`                |
|    5 | FC2 (file → file)                          | `out3.txt` (**file**) | `out4.txt` (**file**) | `w04.txt`, `b04.txt` (**files**) | `V = noodle_fcn("out3.txt", V, 84, "out4.txt", fcn2, ...)`              |
|    6 | FC3 + Softmax (file → RAM)                 | `out4.txt` (**file**) | `BUFFER2` (**var**)   | `w05.txt`, `b05.txt` (**files**) | `V = noodle_fcn("out4.txt", V, 10, BUFFER2, fcn3, ...)`                 |
