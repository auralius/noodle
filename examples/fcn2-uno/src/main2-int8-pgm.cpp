// Noodle INT8 / Uno R3: exact 256->64->10 TFLite parameters in PROGMEM.
// Host protocol matches the file-backed Case-1 Noodle benchmark.
// Keep include/model_quant.h identical to the validated SD Noodle build.
#include <Arduino.h>
#include <math.h>
#include "noodle.h"
#include "noodle_int8.h"
#include "model_quant.h"
#include "model_progmem.h"

#ifndef NOODLE_USE_INT8
#error "Noodle INT8 must be enabled across all compilation units"
#endif
#if !defined(NOODLE_USE_NONE)
#error "Noodle PROGMEM comparison must disable filesystem support"
#endif

static constexpr uint16_t INPUT_N = 256;
static constexpr uint16_t HIDDEN_N = 64;
static constexpr uint16_t OUTPUT_N = 10;
static constexpr uint32_t RX_TIMEOUT_MS = 3000;
static NoodleTensor inputTensor, hiddenTensor, logitsTensor;

extern int __heap_start;
extern void *__brkval;
static int free_ram_bytes() {
  int local;
  const uintptr_t heap_end = __brkval ? (uintptr_t)__brkval : (uintptr_t)&__heap_start;
  return (int)((uintptr_t)&local - heap_end);
}

static void print_mem(const __FlashStringHelper *tag) {
  Serial.print(F("MEM ")); Serial.print(tag);
  Serial.print(F(" I_bytes=")); Serial.print(noodle_tensor_capacity_bytes(&inputTensor));
  Serial.print(F(" H_bytes=")); Serial.print(noodle_tensor_capacity_bytes(&hiddenTensor));
  Serial.print(F(" O_bytes=")); Serial.print(noodle_tensor_capacity_bytes(&logitsTensor));
  Serial.print(F(" arena_used=")); Serial.print(noodle_buffer_arena_used_bytes());
  Serial.print(F(" free_ram=")); Serial.println(free_ram_bytes());
}

static bool receive_bytes(uint8_t *dst, size_t len) {
  const uint32_t started = millis();
  size_t have=0;
  while (have < len) {
    if ((uint32_t)(millis()-started)>RX_TIMEOUT_MS) return false;
    int available=Serial.available();
    if (available<=0) { delay(1); continue; }
    const size_t n = min((size_t)available, len-have);
    int nr=Serial.readBytes((char*)(dst+have),n);
    if (nr>0) have+=(size_t)nr;
  }
  return true;
}

static void quantize_input_in_place() {
  uint8_t *const bytes=(uint8_t*)noodle_tensor_data(&inputTensor);
  for (uint16_t i=0;i<INPUT_N;++i) {
    const uint8_t pix=bytes[i];
    bytes[i]=(uint8_t)(int8_t)noodle_quantize_float(
      (float)pix, noodle_export_i8::layer01_input_scale,
      noodle_export_i8::layer01_input_zero_point);
  }
}

static FCNProgmem layer1() {
  FCNProgmem layer;
  layer.weight_far=NOODLE_PGM_ADDRESS(noodle_w01_flash);
  layer.bias_far=NOODLE_PGM_ADDRESS(noodle_bias01_flash);
  layer.multiplier_far=NOODLE_PGM_ADDRESS(noodle_mult01_flash);
  layer.shift_far=NOODLE_PGM_ADDRESS(noodle_shift01_flash);
  layer.O=HIDDEN_N;
  layer.act=ACT_RELU;
  layer.input_scale=noodle_export_i8::layer01_input_scale;
  layer.input_zero_point=noodle_export_i8::layer01_input_zero_point;
  layer.output_scale=noodle_export_i8::layer01_output_scale;
  layer.output_zero_point=noodle_export_i8::layer01_output_zero_point;
  layer.activation_min=noodle_export_i8::layer01_activation_min;
  layer.activation_max=noodle_export_i8::layer01_activation_max;
  return layer;
}

static FCNProgmem layer2() {
  FCNProgmem layer;
  layer.weight_far=NOODLE_PGM_ADDRESS(noodle_w02_flash);
  layer.bias_far=NOODLE_PGM_ADDRESS(noodle_bias02_flash);
  layer.multiplier_far=NOODLE_PGM_ADDRESS(noodle_mult02_flash);
  layer.shift_far=NOODLE_PGM_ADDRESS(noodle_shift02_flash);
  layer.O=OUTPUT_N;
  layer.act=ACT_NONE;
  layer.input_scale=noodle_export_i8::layer02_input_scale;
  layer.input_zero_point=noodle_export_i8::layer02_input_zero_point;
  layer.output_scale=noodle_export_i8::layer02_output_scale;
  layer.output_zero_point=noodle_export_i8::layer02_output_zero_point;
  layer.activation_min=noodle_export_i8::layer02_activation_min;
  layer.activation_max=noodle_export_i8::layer02_activation_max;
  return layer;
}

static float confidence_from_logits(int8_t top_q) {
  const int8_t *const logits=noodle_tensor_const_data(&logitsTensor);
  const float scale=logitsTensor.scale;
  float sum=0;
  for (uint8_t i=0;i<OUTPUT_N;++i)
    sum += expf(((int16_t)logits[i]-(int16_t)top_q)*scale);
  return sum > 0 ? 1.0f/sum : 0.0f;
}

static bool infer(uint32_t started_us) {
  uint16_t n;
  { const FCNProgmem first=layer1(); n=noodle_fcn(&inputTensor,&hiddenTensor,first); }
  if (n!=HIDDEN_N) { Serial.print(F("ERR FCN1 n=")); Serial.println(n); return false; }
  { const FCNProgmem second=layer2(); n=noodle_fcn(&hiddenTensor,&logitsTensor,second); }
  if (n!=OUTPUT_N) { Serial.print(F("ERR FCN2 n=")); Serial.println(n); return false; }
  uint16_t label=0;
  int8_t max_q=-128;
  noodle_i8_find_max(&logitsTensor,&label,&max_q);
  const float confidence=confidence_from_logits(max_q);
  // Same timing boundaries as Noodle SD Case 1: quantize, FCN1, FCN2, softmax;
  // no serial transmission, no image RX, no model file preflight.
  const uint32_t elapsed_us=micros()-started_us;
  const int8_t *const q=noodle_tensor_const_data(&logitsTensor);
  Serial.print(F("DBG LOGITS_Q"));
  for (uint8_t i=0;i<OUTPUT_N;++i) { Serial.print(' '); Serial.print((int)q[i]); }
  Serial.println();
  Serial.print(F("PRED ")); Serial.print(label);
  Serial.print(' '); Serial.print(elapsed_us*1.0e-6f,4);
  Serial.print(' '); Serial.println(confidence,6);
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  delay(300);
  while(Serial.available()) Serial.read();
  noodle_tensor_init(&inputTensor);
  noodle_tensor_init(&hiddenTensor);
  noodle_tensor_init(&logitsTensor);
  if (!noodle_i8_tensor_require_vector(&inputTensor,INPUT_N,
      noodle_export_i8::layer01_input_scale,
      noodle_export_i8::layer01_input_zero_point)) {
    Serial.println(F("ERR INPUT_ALLOC")); while(true) delay(1000);
  }
  if (!noodle_i8_tensor_require_vector(&hiddenTensor,HIDDEN_N,
      noodle_export_i8::layer01_output_scale,
      noodle_export_i8::layer01_output_zero_point)) {
    Serial.println(F("ERR HIDDEN_ALLOC")); while(true) delay(1000);
  }
  if (!noodle_i8_tensor_require_vector(&logitsTensor,OUTPUT_N,
      noodle_export_i8::layer02_output_scale,
      noodle_export_i8::layer02_output_zero_point)) {
    Serial.println(F("ERR LOGITS_ALLOC")); while(true) delay(1000);
  }
  if(noodle_buffer_arena_used_bytes() != (INPUT_N+HIDDEN_N+OUTPUT_N)) {
    Serial.println(F("ERR ARENA_SIZE")); while(true) delay(1000);
  }
  Serial.println(F("BOOT UNO_FCN16_INT8_PROGMEM"));
  print_mem(F("boot"));
  Serial.println(F("READY"));
}

void loop() {
  if(!Serial.available()) return;
  uint8_t *const input_bytes=(uint8_t*)noodle_tensor_data(&inputTensor);
  if(!input_bytes) { Serial.println(F("ERR INPUT")); return; }
  if (!receive_bytes(input_bytes, INPUT_N)) {
    Serial.println(F("ERR RX_TIMEOUT")); Serial.println(F("READY"));return;
  }
  const uint32_t started_us=micros();
  quantize_input_in_place();
  if (infer(started_us)) print_mem(F("after_predict"));
  Serial.println(F("READY"));
}
