# Anomaly Detection on ESP32
## Preface

- **Author:**  
	- Auralius Manurung (auralius.manurung@ieee.org)  
- **Repositories:** 
	- [Download the whole project files here (Visual Code and PlatformIO).](https://drive.google.com/file/d/1od4ztXoW8vuG_Chqs0oaFd7L5IHgriXM/view?usp=sharing)
	- [Google Colab link.](https://colab.research.google.com/drive/1DLk1hxnl8GVVHyxxxnE-STnxQ50AkTLY?usp=sharing) This is used to extract stored weights and biases from `ad01_fp32.tflite`.
	- [Weights, biases and test datasets.](https://drive.google.com/file/d/1738ZNQD_WDJlxzYc_VnPcVB_vi7KO9H1/view?usp=sharing) These are extracted from the provided `ad01_fp32.tflite`.

## MLPerf Tiny anomaly detection model
In this section, we will implement [the anomali detection which is one of benchmark found in MLPerf Tiny.](https://github.com/mlcommons/tiny.git) The network is entirely made of dense layers with ReLU activations (*Auto-encoder Fully-Connected* or *AEFC*). 

| Layer # | Type  | Input dim | Output dim | Activation    | Extracted weights and biases |
| ------: | ----- | --------: | ---------: | ------------- | ---------------------------- |
|       1 | Dense |       640 |        128 | ReLU          | `w01.txt`, `w02.txt`         |
|       2 | Dense |       128 |        128 | ReLU          | `w03.txt`, `w04.txt`         |
|       3 | Dense |       128 |        128 | ReLU          | `w05.txt`, `w06.txt`         |
|       4 | Dense |       128 |        128 | ReLU          | `w07.txt`, `w08.txt`         |
|       5 | Dense |       128 |          8 | ReLU          | `w09.txt`, `w10.txt`         |
|       6 | Dense |         8 |        128 | ReLU          | `w11.txt`, `w12.txt`         |
|       7 | Dense |       128 |        128 | ReLU          | `w13.txt`, `w14.txt`         |
|       8 | Dense |       128 |        128 | ReLU          | `w15.txt`, `w16.txt`         |
|       9 | Dense |       128 |        128 | ReLU          | `w17.txt`, `w18.txt`         |
|      10 | Dense |       128 |        640 | None (Linear) | `w19.txt`, `w20.txt`         |

## Extracting the weights and biases
The repository provides weights and biases that we can use directly. For this purpose, we selected the trained network stored in `ad01_fp32.tflite`.

## Generating `.txt` from `.wav` files

For our benchmark test, we perform the same preprocessing pipeline used during training ([as found in this part of the repository](https://github.com/mlcommons/tiny/tree/master/benchmark/training/anomaly_detection)). However, we will run the test offline and with viewer data. The ESP32 never processes raw audio. It only consumes precomputed feature vectors stored as text files. Each input `.wav` file is converted into multiple fixed-length feature vectors using the following steps:

- Load WAV file: the audio file is loaded using its native sampling rate (no resampling). The WAV file is 11 seconds with 342 frames.
- Feature extraction: a log-mel spectrogram is computed using the same parameters defined in `baseline.yaml`:
	- `n_mels = 128`
	- `frames = 5`
	- `n_fft = 1024`
	- `hop_length = 512`
	- `power = 2.0`
- Temporal cropping: only the central portion of the spectrogram is kept: `frames 50 to 250 → 200` frames total.
- Sliding window segmentation: a sliding window of length frames = 5 is applied across the cropped spectrogram, producing: `200 − 5 + 1 = 196` feature vectors per WAV file.
- Flattening and storage: each window is flattened into a 1-D vector of size: `inputDim = n_mels × frames = 128 × 5 = 640` and stored as a `float32` text file:
```
<wav_name>_part000.txt
<wav_name>_part001.txt
…
<wav_name>_part195.txt
```

- Take 5 parts from the an anomaly set and name them `anom1.txt` to `anom5.txt`. 
- Take 5 parts from the a normal set and name them `norm1.txt` to `norm5.txt`. 

These `.txt` files represent the actual inputs to the auto-encoder and match exactly the data format used during training and evaluation in the original baseline implementation.

## ESP32 inference workflow

On the ESP32, we will perform the following inference procedures:

- For a given WAV sample, all corresponding `*_partXXX.bin` files are loaded sequentially from SD card (or FFAT).
- Each .bin file is read into a `float[640]` input buffer.
- The input vector is passed through the auto-encoder implemented using Noodle, producing a reconstructed output vector of the same size.
- The mean squared reconstruction error (MSE) between input and output is computed for that window.
- Errors are accumulated across all 196 windows.
- The final anomaly score for the WAV file is computed as the average reconstruction error: `score = mean(MSE_part_0 … MSE_part_195)`

We will only retain the final scalar score and discard the individually reconstructed vectors immediately to minimize memory usage.
## Hardware
For this benchmark, we will use _ESP32-S3-N16R8_ which gives us plenty of space in the flash.
## Testing scenario

- Apply 5 parts from an anomaly dataset ➜ 5/196 of a full WAV.
- Apply 5 parts from a normal dataset ➜ 5/196 of a full WAV.
- Each part is `float32[640]` values (2560 bytes).
- ESP32 returns `mse` and elapsed time (`us`) for each part.

## Code on the ESP side
```cpp
static constexpr uint16_t INPUT_DIM = 640;
static constexpr uint16_t HIDDEN_DIM = 128;
static constexpr uint16_t BOTTLENECK_DIM = 8;

// Ping-pong buffers
static float BUF1[INPUT_DIM];
static float BUF2[INPUT_DIM];

// Copy of input for MSE
static float X0[INPUT_DIM];

FCNFile L1;  
L1.weight_fn  = "/w01.txt"; L1.bias_fn  = "/w02.txt"; L1.act  = ACT_RELU;

FCNFile L2;  
L2.weight_fn  = "/w03.txt"; L2.bias_fn  = "/w04.txt"; L2.act  = ACT_RELU;

FCNFile L3;  
L3.weight_fn  = "/w05.txt"; L3.bias_fn  = "/w06.txt"; L3.act  = ACT_RELU;

FCNFile L4;  
L4.weight_fn  = "/w07.txt"; L4.bias_fn  = "/w08.txt"; L4.act  = ACT_RELU;

FCNFile L5;  
L5.weight_fn  = "/w09.txt"; L5.bias_fn  = "/w10.txt"; L5.act  = ACT_RELU;

FCNFile L6;  
L6.weight_fn  = "/w11.txt"; L6.bias_fn  = "/w12.txt"; L6.act  = ACT_RELU;

FCNFile L7;  
L7.weight_fn  = "/w13.txt"; L7.bias_fn  = "/w14.txt"; L7.act  = ACT_RELU;

FCNFile L8;  
L8.weight_fn  = "/w15.txt"; L8.bias_fn  = "/w16.txt"; L8.act  = ACT_RELU;

FCNFile L9;  
L9.weight_fn  = "/w17.txt"; L9.bias_fn  = "/w18.txt"; L9.act  = ACT_RELU;

FCNFile L10; 
L10.weight_fn = "/w19.txt"; L10.bias_fn = "/w20.txt"; L10.act = ACT_NONE;

uint16_t V = INPUT_DIM;

// 640 -> 128 -> 128 -> 128 -> 128 -> 8 -> 128 -> 128 -> 128 -> 128 -> 640
V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L1,  NULL);
V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L2,  NULL);
V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L3,  NULL);
V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L4,  NULL);
V = noodle_fcn(BUF1, V, BOTTLENECK_DIM,  BUF2, L5,  NULL);
V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L6,  NULL);
V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L7,  NULL);
V = noodle_fcn(BUF2, V, HIDDEN_DIM,      BUF1, L8,  NULL);
V = noodle_fcn(BUF1, V, HIDDEN_DIM,      BUF2, L9,  NULL);
V = noodle_fcn(BUF2, V, INPUT_DIM,       BUF1, L10, NULL);
```
## Compare ESP32 and Google Colab
__ESP 32__
```
=== anom set ===
anom1: mse=9.17803478 us=23812318
anom2: mse=8.465662 us=23812164
anom3: mse=8.66184902 us=23812145
anom4: mse=9.07823086 us=23812145
anom5: mse=8.99418163 us=23812144
Mean anom MSE = 8.87559128

=== norm set ===
norm1: mse=10.8290482 us=23812112
norm2: mse=11.2695885 us=23812183
norm3: mse=11.530509 us=23812167
norm4: mse=11.6987247 us=23812152
norm5: mse=11.0928583 us=23812148
Mean norm MSE = 11.2841454

DONE (processed anom1..5 + norm1..5)
```

__Google Colab__

- [Google Colab link](https://colab.research.google.com/drive/1DLk1hxnl8GVVHyxxxnE-STnxQ50AkTLY?usp=sharing)
- [Weights, biases and test datasets](https://drive.google.com/file/d/1738ZNQD_WDJlxzYc_VnPcVB_vi7KO9H1/view?usp=sharing)

```
Input : input_1 shape= [  1 640] dtype= <class 'numpy.float32'>
Output: Identity shape= [  1 640] dtype= <class 'numpy.float32'>

--- ANOM ---
[TFLite] /content/sample_data/anom1.txt: mse=9.17908482
[TFLite] /content/sample_data/anom2.txt: mse=8.46575602
[TFLite] /content/sample_data/anom3.txt: mse=8.6624254
[TFLite] /content/sample_data/anom4.txt: mse=9.07872793
[TFLite] /content/sample_data/anom5.txt: mse=8.99621678

--- NORM ---
[TFLite] /content/sample_data/norm1.txt: mse=10.82921
[TFLite] /content/sample_data/norm2.txt: mse=11.2699071
[TFLite] /content/sample_data/norm3.txt: mse=11.5304345
[TFLite] /content/sample_data/norm4.txt: mse=11.6980121
[TFLite] /content/sample_data/norm5.txt: mse=11.0927933

--- SUMMARY ---
norm mean=11.2840714 std=0.344865398 min=10.82921 max=11.6980121
anom mean=8.87644219 std=0.300551306 min=8.46575602 max=9.17908482
anom/norm mean ratio = 0.786634706

Done.
```

## Inference parity: ESP32 vs Colab

### Anomaly set

| Sample      | ESP32 MSE | TFLite MSE | Δ       |
| ----------- | --------- | ---------- | ------- |
| `anom1.txt` | 9.1780    | 9.1791     | ~0.001  |
| `anom2.txt` | 8.4657    | 8.4658     | ~0.0001 |
| `anom3.txt` | 8.6618    | 8.6624     | ~0.0006 |
| `anom4.txt` | 9.0782    | 9.0787     | ~0.0005 |
| `anom5.txt` | 8.9942    | 8.9962     | ~0.002  |

### Normal set

| Sample      | ESP32 MSE | TFLite MSE | Δ (abs)   |
| ----------- | --------- | ---------- | --------- |
| `norm1.txt` | 10.82897  | 10.82921   | ≈ 0.00024 |
| `norm2.txt` | 11.27035  | 11.26991   | ≈ 0.00044 |
| `norm3.txt` | 11.53070  | 11.53043   | ≈ 0.00027 |
| `norm4.txt` | 11.69870  | 11.69801   | ≈ 0.00069 |
| `norm5.txt` | 11.09396  | 11.09279   | ≈ 0.00117 |
