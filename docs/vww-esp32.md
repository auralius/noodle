# Visual Wake Word on ESP 32
## Preface

- **Author:**  
	- Auralius Manurung (auralius.manurung@ieee.org)  
- **Repositories:** 
	- [GitHub (Visual Code and PlatformIO)](https://github.com/auralius/noodle/tree/main/examples/mlperf-vww-esp32).

This example is based on the [MLPerf Tiny Visual Wake Word (VWW) benchmark](https://github.com/mlcommons/tiny/tree/master/benchmark/evaluation/datasets) by using [COCO2014 dataset](https://drive.google.com/file/d/1svRAHzb3PeeHG3o8FlfJwSE1aDpqSHe6/view?usp=sharing). The classification task in to predict if there is person(s) in the image (the “person vs non-person” benchmark). The paper can be found [here](https://arxiv.org/pdf/2106.07597).
## MobileNetV1 
The architecture is based on the MLPerf Tiny paper, which is slightly different than the original [MobileNetV1 paper](https://arxiv.org/abs/1704.04861).

| Layer # | Type                         |  S  |  P  | Weight file (dims)          | Bias file (dims)    |
| ------: | ---------------------------- | :-: | :-: | --------------------------- | ------------------- |
|      01 | Conv 3×3 (Cin=3, Cout=8)     |  2  |  1  | `w01.txt` (**3×3×3×8**)     | `b01.txt` (**8**)   |
|      02 | Depthwise 3×3 (Cin=8, M=1)   |  1  |  1  | `w02.txt` (**3×3×8**)       | `b02.txt` (**8**)   |
|      03 | Conv 1×1 (Cin=8, Cout=16)    |  1  |  0  | `w03.txt` (**1×1×8×16**)    | `b03.txt` (**16**)  |
|      04 | Depthwise 3×3 (Cin=16, M=1)  |  2  |  1  | `w04.txt` (**3×3×16**)      | `b04.txt` (**16**)  |
|      05 | Conv 1×1 (Cin=16, Cout=32)   |  1  |  0  | `w05.txt` (**1×1×16×32**)   | `b05.txt` (**32**)  |
|      06 | Depthwise 3×3 (Cin=32, M=1)  |  1  |  1  | `w06.txt` (**3×3×32**)      | `b06.txt` (**32**)  |
|      07 | Conv 1×1 (Cin=32, Cout=32)   |  1  |  0  | `w07.txt` (**1×1×32×32**)   | `b07.txt` (**32**)  |
|      08 | Depthwise 3×3 (Cin=32, M=1)  |  2  |  1  | `w08.txt` (**3×3×32**)      | `b08.txt` (**32**)  |
|      09 | Conv 1×1 (Cin=32, Cout=64)   |  1  |  0  | `w09.txt` (**1×1×32×64**)   | `b09.txt` (**64**)  |
|      10 | Depthwise 3×3 (Cin=64, M=1)  |  1  |  1  | `w10.txt` (**3×3×64**)      | `b10.txt` (**64**)  |
|      11 | Conv 1×1 (Cin=64, Cout=64)   |  1  |  0  | `w11.txt` (**1×1×64×64**)   | `b11.txt` (**64**)  |
|      12 | Depthwise 3×3 (Cin=64, M=1)  |  2  |  1  | `w12.txt` (**3×3×64**)      | `b12.txt` (**64**)  |
|      13 | Conv 1×1 (Cin=64, Cout=128)  |  1  |  0  | `w13.txt` (**1×1×64×128**)  | `b13.txt` (**128**) |
|      14 | Depthwise 3×3 (Cin=128, M=1) |  1  |  1  | `w14.txt` (**3×3×128**)     | `b14.txt` (**128**) |
|      15 | Conv 1×1 (Cin=128, Cout=128) |  1  |  0  | `w15.txt` (**1×1×128×128**) | `b15.txt` (**128**) |
|      16 | Depthwise 3×3 (Cin=128, M=1) |  1  |  1  | `w16.txt` (**3×3×128**)     | `b16.txt` (**128**) |
|      17 | Conv 1×1 (Cin=128, Cout=128) |  1  |  0  | `w17.txt` (**1×1×128×128**) | `b17.txt` (**128**) |
|      18 | Depthwise 3×3 (Cin=128, M=1) |  1  |  1  | `w18.txt` (**3×3×128**)     | `b18.txt` (**128**) |
|      19 | Conv 1×1 (Cin=128, Cout=128) |  1  |  0  | `w19.txt` (**1×1×128×128**) | `b19.txt` (**128**) |
|      20 | Depthwise 3×3 (Cin=128, M=1) |  1  |  1  | `w20.txt` (**3×3×128**)     | `b20.txt` (**128**) |
|      21 | Conv 1×1 (Cin=128, Cout=128) |  1  |  0  | `w21.txt` (**1×1×128×128**) | `b21.txt` (**128**) |
|      22 | Depthwise 3×3 (Cin=128, M=1) |  1  |  1  | `w22.txt` (**3×3×128**)     | `b22.txt` (**128**) |
|      23 | Conv 1×1 (Cin=128, Cout=128) |  1  |  0  | `w23.txt` (**1×1×128×128**) | `b23.txt` (**128**) |
|      24 | Depthwise 3×3 (Cin=128, M=1) |  2  |  1  | `w24.txt` (**3×3×128**)     | `b24.txt` (**128**) |
|      25 | Conv 1×1 (Cin=128, Cout=256) |  1  |  0  | `w25.txt` (**1×1×128×256**) | `b25.txt` (**256**) |
|      26 | Depthwise 3×3 (Cin=256, M=1) |  1  |  1  | `w26.txt` (**3×3×256**)     | `b26.txt` (**256**) |
|      27 | Conv 1×1 (Cin=256, Cout=256) |  1  |  0  | `w27.txt` (**1×1×256×256**) | `b27.txt` (**256**) |
|      28 | FC (in=256, out=2)           |  –  |  –  | `w28.txt` (**2×256**)       | `b28.txt` (**2**)   |
```
Op 00: CONV_2D
  inputs : [(0, [1, 96, 96, 3]), (44, [8, 3, 3, 3]), (3, [8])]
  outputs: [(58, [1, 48, 48, 8])]
  stride : (2, 2)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 01: DEPTHWISE_CONV_2D
  inputs : [(58, [1, 48, 48, 8]), (5, [1, 3, 3, 8]), (4, [8])]
  outputs: [(59, [1, 48, 48, 8])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 02: CONV_2D
  inputs : [(59, [1, 48, 48, 8]), (45, [16, 1, 1, 8]), (21, [16])]
  outputs: [(60, [1, 48, 48, 16])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 03: DEPTHWISE_CONV_2D
  inputs : [(60, [1, 48, 48, 16]), (33, [1, 3, 3, 16]), (32, [16])]
  outputs: [(61, [1, 24, 24, 16])]
  stride : (2, 2)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 04: CONV_2D
  inputs : [(61, [1, 24, 24, 16]), (46, [32, 1, 1, 16]), (34, [32])]
  outputs: [(62, [1, 24, 24, 32])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 05: DEPTHWISE_CONV_2D
  inputs : [(62, [1, 24, 24, 32]), (36, [1, 3, 3, 32]), (35, [32])]
  outputs: [(63, [1, 24, 24, 32])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 06: CONV_2D
  inputs : [(63, [1, 24, 24, 32]), (47, [32, 1, 1, 32]), (37, [32])]
  outputs: [(64, [1, 24, 24, 32])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 07: DEPTHWISE_CONV_2D
  inputs : [(64, [1, 24, 24, 32]), (39, [1, 3, 3, 32]), (38, [32])]
  outputs: [(65, [1, 12, 12, 32])]
  stride : (2, 2)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 08: CONV_2D
  inputs : [(65, [1, 12, 12, 32]), (48, [64, 1, 1, 32]), (40, [64])]
  outputs: [(66, [1, 12, 12, 64])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 09: DEPTHWISE_CONV_2D
  inputs : [(66, [1, 12, 12, 64]), (42, [1, 3, 3, 64]), (41, [64])]
  outputs: [(67, [1, 12, 12, 64])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 10: CONV_2D
  inputs : [(67, [1, 12, 12, 64]), (49, [64, 1, 1, 64]), (6, [64])]
  outputs: [(68, [1, 12, 12, 64])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 11: DEPTHWISE_CONV_2D
  inputs : [(68, [1, 12, 12, 64]), (8, [1, 3, 3, 64]), (7, [64])]
  outputs: [(69, [1, 6, 6, 64])]
  stride : (2, 2)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 12: CONV_2D
  inputs : [(69, [1, 6, 6, 64]), (50, [128, 1, 1, 64]), (9, [128])]
  outputs: [(70, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 13: DEPTHWISE_CONV_2D
  inputs : [(70, [1, 6, 6, 128]), (11, [1, 3, 3, 128]), (10, [128])]
  outputs: [(71, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 14: CONV_2D
  inputs : [(71, [1, 6, 6, 128]), (51, [128, 1, 1, 128]), (12, [128])]
  outputs: [(72, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 15: DEPTHWISE_CONV_2D
  inputs : [(72, [1, 6, 6, 128]), (14, [1, 3, 3, 128]), (13, [128])]
  outputs: [(73, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 16: CONV_2D
  inputs : [(73, [1, 6, 6, 128]), (52, [128, 1, 1, 128]), (15, [128])]
  outputs: [(74, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 17: DEPTHWISE_CONV_2D
  inputs : [(74, [1, 6, 6, 128]), (17, [1, 3, 3, 128]), (16, [128])]
  outputs: [(75, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 18: CONV_2D
  inputs : [(75, [1, 6, 6, 128]), (53, [128, 1, 1, 128]), (18, [128])]
  outputs: [(76, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 19: DEPTHWISE_CONV_2D
  inputs : [(76, [1, 6, 6, 128]), (20, [1, 3, 3, 128]), (19, [128])]
  outputs: [(77, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 20: CONV_2D
  inputs : [(77, [1, 6, 6, 128]), (54, [128, 1, 1, 128]), (22, [128])]
  outputs: [(78, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 21: DEPTHWISE_CONV_2D
  inputs : [(78, [1, 6, 6, 128]), (24, [1, 3, 3, 128]), (23, [128])]
  outputs: [(79, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 22: CONV_2D
  inputs : [(79, [1, 6, 6, 128]), (55, [128, 1, 1, 128]), (25, [128])]
  outputs: [(80, [1, 6, 6, 128])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 23: DEPTHWISE_CONV_2D
  inputs : [(80, [1, 6, 6, 128]), (27, [1, 3, 3, 128]), (26, [128])]
  outputs: [(81, [1, 3, 3, 128])]
  stride : (2, 2)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 24: CONV_2D
  inputs : [(81, [1, 3, 3, 128]), (56, [256, 1, 1, 128]), (28, [256])]
  outputs: [(82, [1, 3, 3, 256])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 25: DEPTHWISE_CONV_2D
  inputs : [(82, [1, 3, 3, 256]), (30, [1, 3, 3, 256]), (29, [256])]
  outputs: [(83, [1, 3, 3, 256])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  depth_multiplier: 1
  act    : 1

Op 26: CONV_2D
  inputs : [(83, [1, 3, 3, 256]), (57, [256, 1, 1, 256]), (31, [256])]
  outputs: [(84, [1, 3, 3, 256])]
  stride : (1, 1)
  padding: SAME
  dilation: (1, 1)
  act    : 1

Op 27: AVERAGE_POOL_2D
  inputs : [(84, [1, 3, 3, 256])]
  outputs: [(85, [1, 1, 1, 256])]
  stride : (3, 3)
  filter : (3, 3)
  padding: VALID
  act    : 0

Op 28: BUILTIN_22
  inputs : [(85, [1, 1, 1, 256]), (2, [2])]
  outputs: [(86, [1, 256])]

Op 29: FULLY_CONNECTED
  inputs : [(86, [1, 256]), (43, [2, 256]), (1, [2])]
  outputs: [(87, [1, 2])]

Op 30: BUILTIN_25
  inputs : [(87, [1, 2])]
  outputs: [(88, [1, 2])]
```
## Extract benchmark data
MLPerf Tiny has already provided `.tflite` file. Therefore, we will first extract all weights and biases by using [the following python code](https://drive.google.com/file/d/15lfleTMr7B73JQnjhnBMJ1q62nC9ilJr/view?usp=sharing). We then put all these files to `data`  folder of the VS Code / PlatformIO  working directory.

## Layer-by-Layer Implementation on ESP32

The following C++ code demonstrates the "Noodle" layer-by-layer implementation for the ESP32:

```cpp
Pool POOL_ID; POOL_ID.M = 1; POOL_ID.T = 1;

uint16_t W = IN_W;
uint16_t V = 0;

Conv c00; c00.K=3; c00.P=1; c00.S=2; c00.weight_fn=W_OP00; c00.bias_fn=B_OP00; c00.act=ACT_RELU;
V = noodle_conv_float(IN, 3, 8, A, W, c00, POOL_ID, nullptr); W = V;

Conv d01; d01.K=3; d01.P=1; d01.S=1; d01.weight_fn=W_OP01; d01.bias_fn=B_OP01; d01.act=ACT_RELU;
V = noodle_dwconv_float(A, 8, B, W, d01, POOL_ID, nullptr); W = V;

Conv c02; c02.K=1; c02.P=0; c02.S=1; c02.weight_fn=W_OP02; c02.bias_fn=B_OP02; c02.act=ACT_RELU;
V = noodle_conv_float(B, 8, 16, A, W, c02, POOL_ID, nullptr); W = V;

Conv d03; d03.K=3; d03.P=1; d03.S=2; d03.weight_fn=W_OP03; d03.bias_fn=B_OP03; d03.act=ACT_RELU;
V = noodle_dwconv_float(A, 16, B, W, d03, POOL_ID, nullptr); W = V;

Conv c04; c04.K=1; c04.P=0; c04.S=1; c04.weight_fn=W_OP04; c04.bias_fn=B_OP04; c04.act=ACT_RELU;
V = noodle_conv_float(B, 16, 32, A, W, c04, POOL_ID, nullptr); W = V;

Conv d05; d05.K=3; d05.P=1; d05.S=1; d05.weight_fn=W_OP05; d05.bias_fn=B_OP05; d05.act=ACT_RELU;
V = noodle_dwconv_float(A, 32, B, W, d05, POOL_ID, nullptr); W = V;

Conv c06; c06.K=1; c06.P=0; c06.S=1; c06.weight_fn=W_OP06; c06.bias_fn=B_OP06; c06.act=ACT_RELU;
V = noodle_conv_float(B, 32, 32, A, W, c06, POOL_ID, nullptr); W = V;

Conv d07; d07.K=3; d07.P=1; d07.S=2; d07.weight_fn=W_OP07; d07.bias_fn=B_OP07; d07.act=ACT_RELU;
V = noodle_dwconv_float(A, 32, B, W, d07, POOL_ID, nullptr); W = V;

Conv c08; c08.K=1; c08.P=0; c08.S=1; c08.weight_fn=W_OP08; c08.bias_fn=B_OP08; c08.act=ACT_RELU;
V = noodle_conv_float(B, 32, 64, A, W, c08, POOL_ID, nullptr); W = V;

Conv d09; d09.K=3; d09.P=1; d09.S=1; d09.weight_fn=W_OP09; d09.bias_fn=B_OP09; d09.act=ACT_RELU;
V = noodle_dwconv_float(A, 64, B, W, d09, POOL_ID, nullptr); W = V;

Conv c10; c10.K=1; c10.P=0; c10.S=1; c10.weight_fn=W_OP10; c10.bias_fn=B_OP10; c10.act=ACT_RELU;
V = noodle_conv_float(B, 64, 64, A, W, c10, POOL_ID, nullptr); W = V;

Conv d11; d11.K=3; d11.P=1; d11.S=2; d11.weight_fn=W_OP11; d11.bias_fn=B_OP11; d11.act=ACT_RELU;
V = noodle_dwconv_float(A, 64, B, W, d11, POOL_ID, nullptr); W = V;

Conv c12; c12.K=1; c12.P=0; c12.S=1; c12.weight_fn=W_OP12; c12.bias_fn=B_OP12; c12.act=ACT_RELU;
V = noodle_conv_float(B, 64, 128, A, W, c12, POOL_ID, nullptr); W = V;

Conv d13; d13.K=3; d13.P=1; d13.S=1; d13.weight_fn=W_OP13; d13.bias_fn=B_OP13; d13.act=ACT_RELU;
V = noodle_dwconv_float(A, 128, B, W, d13, POOL_ID, nullptr); W = V;

Conv c14; c14.K=1; c14.P=0; c14.S=1; c14.weight_fn=W_OP14; c14.bias_fn=B_OP14; c14.act=ACT_RELU;
V = noodle_conv_float(B, 128, 128, A, W, c14, POOL_ID, nullptr); W = V;

Conv d15; d15.K=3; d15.P=1; d15.S=1; d15.weight_fn=W_OP15; d15.bias_fn=B_OP15; d15.act=ACT_RELU;
V = noodle_dwconv_float(A, 128, B, W, d15, POOL_ID, nullptr); W = V;

Conv c16; c16.K=1; c16.P=0; c16.S=1; c16.weight_fn=W_OP16; c16.bias_fn=B_OP16; c16.act=ACT_RELU;
V = noodle_conv_float(B, 128, 128, A, W, c16, POOL_ID, nullptr); W = V;

Conv d17; d17.K=3; d17.P=1; d17.S=1; d17.weight_fn=W_OP17; d17.bias_fn=B_OP17; d17.act=ACT_RELU;
V = noodle_dwconv_float(A, 128, B, W, d17, POOL_ID, nullptr); W = V;

Conv c18; c18.K=1; c18.P=0; c18.S=1; c18.weight_fn=W_OP18; c18.bias_fn=B_OP18; c18.act=ACT_RELU;
V = noodle_conv_float(B, 128, 128, A, W, c18, POOL_ID, nullptr); W = V;

Conv d19; d19.K=3; d19.P=1; d19.S=1; d19.weight_fn=W_OP19; d19.bias_fn=B_OP19; d19.act=ACT_RELU;
V = noodle_dwconv_float(A, 128, B, W, d19, POOL_ID, nullptr); W = V;

Conv c20; c20.K=1; c20.P=0; c20.S=1; c20.weight_fn=W_OP20; c20.bias_fn=B_OP20; c20.act=ACT_RELU;
V = noodle_conv_float(B, 128, 128, A, W, c20, POOL_ID, nullptr); W = V;

Conv d21; d21.K=3; d21.P=1; d21.S=1; d21.weight_fn=W_OP21; d21.bias_fn=B_OP21; d21.act=ACT_RELU;
V = noodle_dwconv_float(A, 128, B, W, d21, POOL_ID, nullptr); W = V;

Conv c22; c22.K=1; c22.P=0; c22.S=1; c22.weight_fn=W_OP22; c22.bias_fn=B_OP22; c22.act=ACT_RELU;
V = noodle_conv_float(B, 128, 128, A, W, c22, POOL_ID, nullptr); W = V;

Conv d23; d23.K=3; d23.P=1; d23.S=2; d23.weight_fn=W_OP23; d23.bias_fn=B_OP23; d23.act=ACT_RELU;
V = noodle_dwconv_float(A, 128, B, W, d23, POOL_ID, nullptr); W = V;

Conv c24; c24.K=1; c24.P=0; c24.S=1; c24.weight_fn=W_OP24; c24.bias_fn=B_OP24; c24.act=ACT_RELU;
V = noodle_conv_float(B, 128, 256, A, W, c24, POOL_ID, nullptr); W = V;

Conv d25; d25.K=3; d25.P=1; d25.S=1; d25.weight_fn=W_OP25; d25.bias_fn=B_OP25; d25.act=ACT_RELU;
V = noodle_dwconv_float(A, 256, B, W, d25, POOL_ID, nullptr); W = V;

Conv c26; c26.K=1; c26.P=0; c26.S=1; c26.weight_fn=W_OP26; c26.bias_fn=B_OP26; c26.act=ACT_RELU;
V = noodle_conv_float(B, 256, 256, A, W, c26, POOL_ID, nullptr); W = V;

noodle_gap(A, 256, W);

float out2[2];
FCNFile fcf; fcf.weight_fn = FC_W; fcf.bias_fn = FC_B; fcf.act = ACT_SOFTMAX;
(void)noodle_fcn((const float*)A, 256, 2, out2, fcf, nullptr);

```

## Testing Scenario

1. **Image Processing:** [A Python script](attachments/send_test_data.py) loads an image, converts it into an array, and sends it to the ESP32 via serial communication.
2. **Inference:** The ESP32 receives the data, processes the layers, and calculates the prediction.
3. **Output:** The ESP32 returns the classification results and the total execution time to the host.

```
Found 10 images in 'non_person'
PY DBG first pixel normalized: R=0.094118 G=0.086275 B=0.105882
[1/10] COCO_train2014_000000387393.jpg -> P0=0.964005 P1=0.035995 pred=0 time_ms=29869
[2/10] COCO_train2014_000000395007.jpg -> P0=0.926328 P1=0.073672 pred=0 time_ms=29870
[3/10] COCO_val2014_000000556005.jpg -> P0=0.850917 P1=0.149083 pred=0 time_ms=29870
[4/10] COCO_train2014_000000091234.jpg -> P0=0.931560 P1=0.068440 pred=0 time_ms=29869
[5/10] COCO_train2014_000000257732.jpg -> P0=0.916183 P1=0.083817 pred=0 time_ms=29869
[6/10] COCO_train2014_000000187844.jpg -> P0=0.668761 P1=0.331239 pred=0 time_ms=29870
[7/10] COCO_train2014_000000037880.jpg -> P0=0.960637 P1=0.039363 pred=0 time_ms=29869
[8/10] COCO_val2014_000000326854.jpg -> P0=0.983913 P1=0.016087 pred=0 time_ms=29870
[9/10] COCO_train2014_000000044842.jpg -> P0=0.973420 P1=0.026580 pred=0 time_ms=29871
[10/10] COCO_val2014_000000558362.jpg -> P0=0.774368 P1=0.225632 pred=0 time_ms=29870

Summary: n=10 avg_P1=0.104991 avg_time_ms=29869.7

Found 10 images in 'person'
PY DBG first pixel normalized: R=0.819608 G=0.403922 B=0.301961
[1/10] COCO_val2014_000000533123.jpg -> P0=0.273155 P1=0.726845 pred=1 time_ms=29870
[2/10] COCO_val2014_000000127516.jpg -> P0=0.355667 P1=0.644333 pred=1 time_ms=29870
[3/10] COCO_val2014_000000287741.jpg -> P0=0.862113 P1=0.137887 pred=0 time_ms=29870
[4/10] COCO_train2014_000000523490.jpg -> P0=0.548813 P1=0.451187 pred=0 time_ms=29870
[5/10] COCO_train2014_000000091056.jpg -> P0=0.687100 P1=0.312900 pred=0 time_ms=29870
[6/10] COCO_train2014_000000101218.jpg -> P0=0.336044 P1=0.663956 pred=1 time_ms=29871
[7/10] COCO_train2014_000000169510.jpg -> P0=0.844256 P1=0.155744 pred=0 time_ms=29870
[8/10] COCO_val2014_000000138995.jpg -> P0=0.662459 P1=0.337541 pred=0 time_ms=29870
[9/10] COCO_val2014_000000022793.jpg -> P0=0.911852 P1=0.088148 pred=0 time_ms=29871
[10/10] COCO_train2014_000000336552.jpg -> P0=0.294411 P1=0.705589 pred=1 time_ms=29870

Summary: n=10 avg_P1=0.422413 avg_time_ms=29870.2
```

4. Validation: Compare the results with [Python and Tensorflow](attachments/validation.py) using the provided trained `.tflite`.5. 
```
[1/10] COCO_train2014_000000387393.jpg -> P0=0.974202 P1=0.025798 pred=0
[2/10] COCO_train2014_000000395007.jpg -> P0=0.790333 P1=0.209667 pred=0
[3/10] COCO_val2014_000000556005.jpg -> P0=0.942274 P1=0.057726 pred=0
[4/10] COCO_train2014_000000091234.jpg -> P0=0.797478 P1=0.202522 pred=0
[5/10] COCO_train2014_000000257732.jpg -> P0=0.879772 P1=0.120228 pred=0
[6/10] COCO_train2014_000000187844.jpg -> P0=0.777542 P1=0.222458 pred=0
[7/10] COCO_train2014_000000037880.jpg -> P0=0.977813 P1=0.022187 pred=0
[8/10] COCO_val2014_000000326854.jpg -> P0=0.969205 P1=0.030795 pred=0
[9/10] COCO_train2014_000000044842.jpg -> P0=0.974536 P1=0.025464 pred=0
[10/10] COCO_val2014_000000558362.jpg -> P0=0.896006 P1=0.103994 pred=0

Summary: n=10 avg_P1=0.102084

[1/10] COCO_val2014_000000533123.jpg -> P0=0.001624 P1=0.998376 pred=1
[2/10] COCO_val2014_000000127516.jpg -> P0=0.068915 P1=0.931085 pred=1
[3/10] COCO_val2014_000000287741.jpg -> P0=0.520537 P1=0.479463 pred=0
[4/10] COCO_train2014_000000523490.jpg -> P0=0.063276 P1=0.936723 pred=1
[5/10] COCO_train2014_000000091056.jpg -> P0=0.072531 P1=0.927469 pred=1
[6/10] COCO_train2014_000000101218.jpg -> P0=0.053075 P1=0.946925 pred=1
[7/10] COCO_train2014_000000169510.jpg -> P0=0.385355 P1=0.614645 pred=1
[8/10] COCO_val2014_000000138995.jpg -> P0=0.243333 P1=0.756667 pred=1
[9/10] COCO_val2014_000000022793.jpg -> P0=0.694776 P1=0.305224 pred=0
[10/10] COCO_train2014_000000336552.jpg -> P0=0.005157 P1=0.994843 pred=1

Summary: n=10 avg_P1=0.789142
```

## Remarks
Discrepancies currently exist between TensorFlow and Noodle outputs. Since reusing pre-trained weights and biases requires bit-perfect process parity, we are finding it difficult to trace and align specific execution steps. To ensure consistency moving forward, it may be more efficient to train models from scratch within our own ecosystem and develop a custom model exporter.