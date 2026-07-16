/**
 * Serial FCN-16 benchmark for Arduino Uno R3.
 *
 * Model:
 *   16x16 uint8 image -> FCN 256->64 ReLU -> FCN 64->10 Softmax
 *
 * Parameters:
 *   w01.bin, b01.bin, w02.bin, b02.bin on SD card.
 *
 * Serial protocol:
 *   Wait for READY, then send exactly 256 uint8 pixels.
 *   Uno replies:
 *     PRED <label> <seconds> <confidence>
 *     MEM <fields...>
 *     READY
 */

#include <Arduino.h>
#include <avr/pgmspace.h>
#include "noodle.h"

static const uint32_t BAUD = 115200;
static const uint32_t RX_TIMEOUT_MS = 3000;

#ifndef NOODLE_SD_CS
#define NOODLE_SD_CS 10
#endif

static const uint16_t IMG_SIZE = 16 * 16;
static const uint16_t HIDDEN = 64;
static const uint16_t CLASSES = 10;

#define NORMALIZE_0_1

static uint8_t RX_BYTES[IMG_SIZE];
static NoodleBuffer A;
static NoodleBuffer B;

extern int __heap_start;
extern void *__brkval;

static int free_ram_bytes()
{
  int stack_top;
  const uintptr_t heap_end =
      (__brkval == nullptr) ? (uintptr_t)&__heap_start : (uintptr_t)__brkval;
  return (int)((uintptr_t)&stack_top - heap_end);
}

static void print_memory_state(const __FlashStringHelper *tag)
{
  const size_t arena_capacity = noodle_buffer_arena_capacity_bytes();
  const size_t arena_used = noodle_buffer_arena_used_bytes();
  const size_t arena_headroom =
      (arena_capacity >= arena_used) ? arena_capacity - arena_used : 0;

  const size_t a_bytes = A.capacity * sizeof(float);
  const size_t b_bytes = B.capacity * sizeof(float);

  Serial.print(F("MEM "));
  Serial.print(tag);
  Serial.print(F(" A_cap="));
  Serial.print(A.capacity);
  Serial.print(F(" A_bytes="));
  Serial.print(a_bytes);
  Serial.print(F(" B_cap="));
  Serial.print(B.capacity);
  Serial.print(F(" B_bytes="));
  Serial.print(b_bytes);
  Serial.print(F(" logical_bytes="));
  Serial.print(a_bytes + b_bytes);
  Serial.print(F(" arena_capacity="));
  Serial.print(arena_capacity);
  Serial.print(F(" arena_used="));
  Serial.print(arena_used);
  Serial.print(F(" arena_headroom="));
  Serial.print(arena_headroom);
  Serial.print(F(" free_ram="));
  Serial.println(free_ram_bytes());
}

static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms)
{
  const uint32_t t0 = millis();
  size_t got = 0;

  while (got < n)
  {
    if ((millis() - t0) > timeout_ms)
      return false;

    int avail = Serial.available();
    if (avail <= 0)
    {
      delay(1);
      continue;
    }

    size_t remaining = n - got;
    size_t chunk = ((size_t)avail < remaining) ? (size_t)avail : remaining;
    int r = Serial.readBytes((char *)(dst + got), chunk);
    if (r > 0)
      got += (size_t)r;
  }

  return true;
}

static bool predict()
{
  FCNFile fcn1;
  fcn1.weight_fn = "w01.bin";
  fcn1.bias_fn = "b01.bin";
  fcn1.act = ACT_RELU;

  FCNFile fcn2;
  fcn2.weight_fn = "w02.bin";
  fcn2.bias_fn = "b02.bin";
  fcn2.act = ACT_SOFTMAX;

  const unsigned long t0 = micros();

  uint16_t n = noodle_fcn(
      RX_BYTES, IMG_SIZE, HIDDEN, &A, fcn1, nullptr);

  if (n != HIDDEN)
  {
    Serial.print(F("ERR FCN1 n="));
    Serial.println(n);
    return false;
  }

  n = noodle_fcn(
      &A, n, CLASSES, &B, fcn2, nullptr);

  if (n != CLASSES)
  {
    Serial.print(F("ERR FCN2 n="));
    Serial.println(n);
    return false;
  }

  const unsigned long elapsed_us = micros() - t0;

  uint16_t label = 0;
  float confidence = 0.0f;
  noodle_find_max(&B, CLASSES, confidence, label);

  Serial.print(F("PRED "));
  Serial.print(label);
  Serial.print(' ');
  Serial.print((float)elapsed_us * 1.0e-6f, 4);
  Serial.print(' ');
  Serial.println(confidence, 6);

  return true;
}

void setup()
{
  Serial.begin(BAUD);
  Serial.setTimeout(100);
  delay(300);

  while (Serial.available())
    Serial.read();

  noodle_buffer_init(&A);
  noodle_buffer_init(&B);

  if (!noodle_fs_init(NOODLE_SD_CS))
  {
    Serial.println(F("ERR SD"));
    while (true)
      delay(1000);
  }

  Serial.println(F("BOOT UNO_FCN16_PACKED_ARENA"));
  print_memory_state(F("boot"));
  Serial.println(F("READY"));
}

void loop()
{
  if (!recv_exact(RX_BYTES, IMG_SIZE, RX_TIMEOUT_MS))
  {
    Serial.println(F("READY"));
    return;
  }

  if (predict())
    print_memory_state(F("after_predict"));

  Serial.println(F("READY"));
}
