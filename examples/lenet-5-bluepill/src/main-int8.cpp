/*
 * Noodle Case Study 3 — LeNet-5 full INT8 on STM32F103C8T6 (Blue Pill)
 *
 * Case-1 packed-FCN SD layout: conv1/2 split, FC1/2/3 wNNi.bin + pNNi.bin.
 * Do NOT put FP32 w/b files on the same SD card.
 *
 * Uses the current Noodle INT8 NoodleTensor operators, not custom kernels.
 * Input:  28x28 raw uint8 MNIST, transmitted by existing IMG/ACK sender.
 * Quant: int8_input = uint8_input - 128 (input scale = 1, zp = -128).
 * Shapes: Conv1 SAME 6x28x28 -> avgpool 6x14x14;
 *         Conv2 VALID 16x10x10 -> avgpool 16x5x5;
 *         CHW->HWC flatten 400 -> FC 120 -> FC 84 -> FC 10 -> softmax.
 *
 * Packed pNNi.bin records: bias(int32), multiplier(int32), shift(int32) per neuron.
 * SD: SPI2 PB13=SCK PB14=MISO PB15=MOSI PB12=CS at 3.3V.
 * LED: PC13 active-low.
 *
 * KEEP as a separate build from existing src/main-blue-pill.cpp (FP32).
 */

#include <Arduino.h>
#include <SPI.h>
#include <string.h>
#include <math.h>
#include "noodle.h"
#include "noodle_serial.h"
#include "int8/model_quant.h"   // exact file from INT8 exporter

#ifndef NOODLE_USE_INT8
# error "Build this file with -D NOODLE_USE_INT8"
#endif
#ifndef NOODLE_USE_SDFAT
# error "Case3 Blue Pill needs SdFat"
#endif

namespace {
static const uint8_t SD_CS = PB12;
static const uint8_t LED_PIN = PC13;
static const uint16_t IMG_SIDE = 28;
static const uint16_t IMG_BYTES = IMG_SIDE * IMG_SIDE;
static SPIClass SPI_SD(PB15, PB14, PB13, PB12);
static uint8_t RX_BYTES[IMG_BYTES];

// Expected SD sizes are the sizes exported from the *actual* Case3 model.
static const uint16_t CH[5]    = {6, 16, 120, 84, 10};
static const uint32_t WLEN[5] = {150, 2400, 48000, 10080, 840};
static const char* const WF[5] = {"w01i.bin","w02i.bin","w03i.bin","w04i.bin","w05i.bin"};
static const char* const BF[2] = {"b01i.bin","b02i.bin"};
static const char* const MF[2] = {"m01i.bin","m02i.bin"};
static const char* const SF[2] = {"s01i.bin","s02i.bin"};
static const char* const PF[3] = {"p03i.bin","p04i.bin","p05i.bin"};
static const float INPUT_SCALE[5] = {
    noodle_export_i8::layer01_input_scale,
    noodle_export_i8::layer02_input_scale,
    noodle_export_i8::layer03_input_scale,
    noodle_export_i8::layer04_input_scale,
    noodle_export_i8::layer05_input_scale};
static const float OUTPUT_SCALE[5] = {
    noodle_export_i8::layer01_output_scale,
    noodle_export_i8::layer02_output_scale,
    noodle_export_i8::layer03_output_scale,
    noodle_export_i8::layer04_output_scale,
    noodle_export_i8::layer05_output_scale};
static const int32_t INPUT_ZP[5] = {
    noodle_export_i8::layer01_input_zero_point,
    noodle_export_i8::layer02_input_zero_point,
    noodle_export_i8::layer03_input_zero_point,
    noodle_export_i8::layer04_input_zero_point,
    noodle_export_i8::layer05_input_zero_point};
static const int32_t OUTPUT_ZP[5] = {
    noodle_export_i8::layer01_output_zero_point,
    noodle_export_i8::layer02_output_zero_point,
    noodle_export_i8::layer03_output_zero_point,
    noodle_export_i8::layer04_output_zero_point,
    noodle_export_i8::layer05_output_zero_point};
static const int32_t MIN_ACT[5] = {
    noodle_export_i8::layer01_activation_min,
    noodle_export_i8::layer02_activation_min,
    noodle_export_i8::layer03_activation_min,
    noodle_export_i8::layer04_activation_min,
    noodle_export_i8::layer05_activation_min};
static const int32_t MAX_ACT[5] = {
    noodle_export_i8::layer01_activation_max,
    noodle_export_i8::layer02_activation_max,
    noodle_export_i8::layer03_activation_max,
    noodle_export_i8::layer04_activation_max,
    noodle_export_i8::layer05_activation_max};

static void blink(uint8_t times) {
  for (uint8_t i=0; i<times; ++i) {
    digitalWrite(LED_PIN, LOW); delay(120);
    digitalWrite(LED_PIN, HIGH); delay(120);
  }
}

static void fail_forever(const char* reason) {
  Serial.print(F("ERR BOOT ")); Serial.println(reason);
  Serial.flush();
  while (true) { blink(3); delay(500); }
}

static bool exact_file_size(const char* name, uint32_t expected) {
  NDL_File f = noodle_fs_open_read(name);
  if (!f) {
    Serial.print(F("ERR FILE_MISSING ")); Serial.println(name);
    return false;
  }
  uint32_t size = (uint32_t)f.size();
  f.close();
  if (size != expected) {
    Serial.print(F("ERR FILE_SIZE ")); Serial.print(name);
    Serial.print(' '); Serial.print(size);
    Serial.print(F(" expected=")); Serial.println(expected);
    return false;
  }
  return true;
}

static bool check_sd_files() {
  // Convolution kernels still consume split b/m/s metadata.
  for (uint8_t layer=0; layer<2; ++layer) {
    const uint32_t quant_bytes = (uint32_t)CH[layer] * 4u;
    if (!exact_file_size(WF[layer], WLEN[layer]) ||
        !exact_file_size(BF[layer], quant_bytes) ||
        !exact_file_size(MF[layer], quant_bytes) ||
        !exact_file_size(SF[layer], quant_bytes)) return false;
  }
  // FCN uses the same packed metadata as Noodle Case 1.
  for (uint8_t layer=2; layer<5; ++layer) {
    if (!exact_file_size(WF[layer], WLEN[layer]) ||
        !exact_file_size(PF[layer-2], (uint32_t)CH[layer] * 12u)) return false;
  }
  return true;
}

static uint32_t crc32_pixels(const uint8_t* data, uint16_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (uint16_t i=0; i<n; ++i) {
    crc ^= data[i];
    for (uint8_t b=0; b<8; ++b)
      crc = (crc >> 1) ^ ((crc & 1u) ? 0xEDB88320u : 0u);
  }
  return ~crc;
}

static Conv make_conv(uint8_t layer, uint16_t pad) {
  Conv c;
  c.K = 5;
  c.P = pad;
  c.S = 1;
  c.O = CH[layer];
  c.weight_fn = WF[layer]; c.bias_fn = BF[layer];
  c.multiplier_fn = MF[layer]; c.shift_fn = SF[layer];
  c.input_scale = INPUT_SCALE[layer];
  c.output_scale = OUTPUT_SCALE[layer];
  c.input_zero_point = INPUT_ZP[layer];
  c.output_zero_point = OUTPUT_ZP[layer];
  c.act = ACT_RELU;
  c.activation_min = noodle_activation_min_relu(c.output_scale, c.output_zero_point);
  c.activation_max = MAX_ACT[layer];
  return c;
}

static FCNFile make_fc(uint8_t layer) {
  FCNFile f;
  f.O = CH[layer];
  // Packed metadata is read sequentially per neuron, one open per layer.
  // Do not set split bias_fn / multiplier_fn / shift_fn.
  f.weight_fn = WF[layer];
  f.metadata_fn = PF[layer-2];
  f.input_scale = INPUT_SCALE[layer];
  f.output_scale = OUTPUT_SCALE[layer];
  f.input_zero_point = INPUT_ZP[layer];
  f.output_zero_point = OUTPUT_ZP[layer];
  f.act = (layer < 4) ? ACT_RELU : ACT_NONE;
  f.activation_min = (layer < 4)
      ? noodle_activation_min_relu(f.output_scale, f.output_zero_point)
      : MIN_ACT[layer];
  f.activation_max = MAX_ACT[layer];
  return f;
}

static bool checked_shape(const NoodleTensor& t, uint16_t rank,
                          uint16_t channels, uint16_t width) {
  return t.rank == rank && t.C == channels && t.W == width &&
         noodle_tensor_const_data(&t) != nullptr;
}

static void predict() {
  uint32_t stages[7] = {0,0,0,0,0,0,0};
  NoodleTensor img, p1, p2, flat, f1, f2, logits;
  noodle_tensor_init(&img); noodle_tensor_init(&p1);
  noodle_tensor_init(&p2); noodle_tensor_init(&flat);
  noodle_tensor_init(&f1); noodle_tensor_init(&f2);
  noodle_tensor_init(&logits);

  const uint32_t crc = crc32_pixels(RX_BYTES, IMG_BYTES);
  uint8_t failure = 0;
  uint8_t pred = 0;
  int8_t raw_logits[10] = {0};
  int8_t raw_softmax[10] = {0};
  float confidence = 0.f;
  const uint32_t total_start = micros();

  {
    NoodleData* const raw = noodle_tensor_require_2d(&img, 1, IMG_SIDE);
    if (!raw) { failure=1; goto cleanup; }
    noodle_tensor_set_quantization(&img, 1.0f, -128);
    for (uint16_t i=0; i<IMG_BYTES; ++i)
      raw[i] = (int8_t)((int16_t)RX_BYTES[i] - 128);
  }
  {
    Pool pool; pool.M = 2; pool.T = 2;
    Conv c = make_conv(0, 2);  // SAME -> 28; mean pool -> 14
    uint32_t t0=micros();
    uint16_t width = noodle_conv2d(&img, &p1, c, pool);
    stages[0]=micros()-t0;
    if (width!=14 || !checked_shape(p1,NOODLE_TENSOR_2D,6,14)) {
      failure=2; goto cleanup;
    }
    noodle_tensor_free(&img);
  }
  {
    Pool pool; pool.M = 2; pool.T = 2;
    Conv c = make_conv(1, 0);  // VALID -> 10; mean pool -> 5
    uint32_t t0=micros();
    uint16_t width = noodle_conv2d(&p1, &p2, c, pool);
    stages[1]=micros()-t0;
    if (width!=5 || !checked_shape(p2,NOODLE_TENSOR_2D,16,5)) {
      failure=3; goto cleanup;
    }
    noodle_tensor_free(&p1);
  }
  {
    uint32_t t0=micros();
    uint16_t count = noodle_flat(&p2,&flat); // CHW -> HWC order
    stages[2]=micros()-t0;
    if (count!=400 || !checked_shape(flat, NOODLE_TENSOR_1D,400,1)) {
      failure=4; goto cleanup;
    }
    noodle_tensor_free(&p2);
  }
  {
    FCNFile f=make_fc(2);
    uint32_t t0=micros();
    uint16_t count=noodle_fcn(&flat,&f1,f);
    stages[3]=micros()-t0;
    if (count!=120 || !checked_shape(f1,NOODLE_TENSOR_1D,120,1)) {
      failure=5; goto cleanup;
    }
    noodle_tensor_free(&flat);
  }
  {
    FCNFile f=make_fc(3);
    uint32_t t0=micros();
    uint16_t count=noodle_fcn(&f1,&f2,f);
    stages[4]=micros()-t0;
    if (count!=84 || !checked_shape(f2,NOODLE_TENSOR_1D,84,1)) {
      failure=6; goto cleanup;
    }
    noodle_tensor_free(&f1);
  }
  {
    FCNFile f=make_fc(4);
    uint32_t t0=micros();
    uint16_t count=noodle_fcn(&f2,&logits,f);
    stages[5]=micros()-t0;
    if (count!=10 || !checked_shape(logits,NOODLE_TENSOR_1D,10,1)) {
      failure=7; goto cleanup;
    }
    noodle_tensor_free(&f2);
  }
  {
    const int8_t* q = noodle_tensor_const_data(&logits);
    memcpy(raw_logits,q,10);
    for (uint8_t i=1; i<10; ++i) if (q[i]>q[pred]) pred=i;

    uint32_t t0=micros();
    uint16_t count=noodle_soft_max(&logits);
    stages[6]=micros()-t0;
    if (count!=10 || !checked_shape(logits,NOODLE_TENSOR_1D,10,1)) {
      failure=8; goto cleanup;
    }
    memcpy(raw_softmax,noodle_tensor_const_data(&logits),10);
    // Derive probability from the quantization metadata returned by Noodle,
    // instead of assuming the output zero-point is unchanged from logits.
    confidence = noodle_dequantize_int8(raw_softmax[pred],
                                        logits.scale,logits.zero_point);
  }
cleanup:
  const uint32_t elapsed = micros()-total_start;
  if (failure) {
    Serial.print(F("ERR CASE3_STAGE_")); Serial.println(failure);
  } else {
    Serial.print(F("PRED ")); Serial.print(pred);
    Serial.print(' '); Serial.print(elapsed*1e-6f,4);
    Serial.print(' '); Serial.println(confidence,6);
    Serial.print(F("CRC32 "));
    if (crc<0x10000000u) Serial.print('0');
    if (crc<0x01000000u) Serial.print('0');
    if (crc<0x00100000u) Serial.print('0');
    if (crc<0x00010000u) Serial.print('0');
    if (crc<0x00001000u) Serial.print('0');
    if (crc<0x00000100u) Serial.print('0');
    if (crc<0x00000010u) Serial.print('0');
    Serial.println(crc,HEX);
    Serial.print(F("LOGITS_Q"));
    for (uint8_t i=0; i<10; ++i) { Serial.print(' '); Serial.print((int)raw_logits[i]); }
    Serial.println();
    Serial.print(F("SOFTMAX_Q"));
    for (uint8_t i=0; i<10; ++i) { Serial.print(' '); Serial.print((int)raw_softmax[i]); }
    Serial.println();
    Serial.print(F("SOFTMAX_QUANT "));
    Serial.print(logits.scale,9);
    Serial.print(' '); Serial.println(logits.zero_point);
    static const char* const names[7]={"conv1_pool1","conv2_pool2","flatten","fc1","fc2","fc3","softmax"};
    for(uint8_t i=0; i<7; ++i) {
      Serial.print(F("STAGE ")); Serial.print(names[i]);
      Serial.print(' '); Serial.println(stages[i]); // microseconds
    }
  }
  noodle_tensor_free(&img); noodle_tensor_free(&p1);
  noodle_tensor_free(&p2); noodle_tensor_free(&flat);
  noodle_tensor_free(&f1); noodle_tensor_free(&f2);
  noodle_tensor_free(&logits);
  Serial.flush();
  delay(5);
}
}  // namespace

void setup() {
  pinMode(LED_PIN,OUTPUT);
  digitalWrite(LED_PIN,HIGH);
  blink(2);
  NoodleSerial::begin(115200);
  NoodleSerial::clear_input();
  Serial.println(F("BOOT Blue Pill Noodle LeNet5 FULL INT8"));
  Serial.println(F("INPUT uint8 raw -> int8, scale=1, zp=-128"));
  pinMode(SD_CS,OUTPUT); digitalWrite(SD_CS,HIGH);
  SPI_SD.begin();
  if (!noodle_fs_init(SD_CS,SPI_SD,16)) fail_forever("SD_INIT");
  Serial.println(F("SD mounted OK"));
  if (!check_sd_files()) fail_forever("WRONG_INT8_SD_FILES");
  Serial.println(F("INT8 packed-FCN SD files: 14/14 OK"));
  Serial.println(F("READY"));
}

void loop() {
  if(!NoodleSerial::wait_for_img_header()) {
    NoodleSerial::print_ready();
    return;
  }
  if (!NoodleSerial::recv_image_chunked(RX_BYTES,IMG_BYTES)) {
    NoodleSerial::print_ready();
    return;
  }
  predict();
  NoodleSerial::print_ready();
}
