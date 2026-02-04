/**
 * Auralius Manurung -- Universitas Telkom, Bandung
 *   auralius.manurung@ieee.org
 * Lisa Kristiana -- ITENAS, Bandung
 *   lisa@itenas.ac.id
 */
#include <Arduino.h>
#include "noodle.h"

// Weights/biases generated from Keras PeakNet1D exporter
#include "w01.h"
#include "b01.h"
#include "w02.h"
#include "b02.h"
#include "w03.h"
#include "b03.h"
#include "w04.h"
#include "b04.h"
#include "w05.h"
#include "b05.h"
#include "w06.h"
#include "b06.h"

// -----------------------------
// Signal + model constants
// -----------------------------
static const uint16_t L = 256;      // window length
static const uint16_t MAX_CH = 16;  // PeakNet max channels in mid layers
static const uint32_t BUF_N = (uint32_t)L * (uint32_t)MAX_CH; // 4096 floats

// -----------------------------
// Buffers (CHW packed: [ch0 plane][ch1 plane]...)
// -----------------------------
float *BUFFER1;
float *BUFFER2;
float *BUFFER3;
float *BUFFER4;

// RX frame: "ECG" + 256 float32
static const uint8_t RX_HDR[3] = {'E','C','G'};
static uint8_t RX_RAW[3 + L * 4];

// -----------------------------
// Serial receive helpers
// -----------------------------
static bool recv_exact(uint8_t *dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n) {
    if ((millis() - t0) > timeout_ms) return false;
    int c = Serial.read();
    if (c < 0) { delay(1); continue; }
    dst[got++] = (uint8_t)c;
  }
  return true;
}

static float read_f32_le(const uint8_t *p) {
  // Assumes IEEE754 float and little-endian host (true for most MCUs you’ll use)
  float f;
  memcpy(&f, p, 4);
  return f;
}

// -----------------------------
// Allocate buffers
// -----------------------------
static void alloc_buffers() {
  BUFFER1 = (float*)malloc(L * sizeof(float));
  BUFFER2 = (float*)malloc(L * sizeof(float));

  BUFFER3 = (float*)malloc(BUF_N * sizeof(float));
  BUFFER4 = (float*)malloc(BUF_N * sizeof(float));

  if (!BUFFER1 || !BUFFER2 || !BUFFER3 || !BUFFER4) {
    Serial.println(F("ERROR: malloc failed (out of RAM)"));
    while (true) delay(1000);
  }
}

// -----------------------------
// Run PeakNet1D on BUFFER1 (expects input in BUFFER1[0..255])
// Output: BUFFER1[0..255] = sigmoid scores
// -----------------------------
static void predict_peaknet() {
  unsigned long st = micros();

  // PeakNet1D layers
  ConvMem c1; c1.K=9; c1.P=4; c1.S=1; c1.weight=w01; c1.bias=b01; c1.act=ACT_RELU; // 1->8
  ConvMem c2; c2.K=7; c2.P=3; c2.S=1; c2.weight=w02; c2.bias=b02; c2.act=ACT_RELU; // 8->16
  ConvMem c3; c3.K=7; c3.P=3; c3.S=1; c3.weight=w03; c3.bias=b03; c3.act=ACT_RELU; // 16->16
  ConvMem c4; c4.K=7; c4.P=3; c4.S=1; c4.weight=w04; c4.bias=b04; c4.act=ACT_RELU; // 16->16
  ConvMem c5; c5.K=1; c5.P=0; c5.S=1; c5.weight=w05; c5.bias=b05; c5.act=ACT_RELU; // 16->16
  ConvMem c6; c6.K=1; c6.P=0; c6.S=1; c6.weight=w06; c6.bias=b06; c6.act=ACT_NONE; // 16->1

  uint16_t V = L;

  // noodle_conv1d(in, n_inputs, out, n_outputs, W, conv)
  V = noodle_conv1d(BUFFER3, 1,  BUFFER4, 8,  V, c1, NULL);
  V = noodle_conv1d(BUFFER4, 8,  BUFFER3, 16, V, c2, NULL);
  V = noodle_conv1d(BUFFER3, 16, BUFFER4, 16, V, c3, NULL);
  V = noodle_conv1d(BUFFER4, 16, BUFFER3, 16, V, c4, NULL);
  V = noodle_conv1d(BUFFER3, 16, BUFFER4, 16, V, c5, NULL);
  V = noodle_conv1d(BUFFER4, 16, BUFFER3, 1,  V, c6, NULL);

  // Final sigmoid on 1 channel output
  noodle_sigmoid(BUFFER3, V);

  float et = (float)(micros() - st) * 1e-6f;

  // Send results: SCORES <t_sec> <256 floats...>
  Serial.print(F("SCORES "));
  Serial.print(et, 6);
  for (uint16_t i = 0; i < V; i++) {
    Serial.print(' ');
    Serial.print(BUFFER3[i], 6);
  }
  Serial.println();
}

// -----------------------------
// Arduino setup/loop
// -----------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(2);

  alloc_buffers();
  noodle_setup_temp_buffers(BUFFER1, BUFFER2);
  Serial.println(F("READY")); // Python can wait for this
}

void loop() {
  // Wait for a full frame: "ECG" + 256 float32
  // Timeout can be generous because your host might pause.
  if (!recv_exact(RX_RAW, sizeof(RX_RAW), 10000)) {
    Serial.println(F("READY")); // indicate ready again
    return; // no full frame yet
  }

  // Check header
  if (RX_RAW[0] != RX_HDR[0] || RX_RAW[1] != RX_HDR[1] || RX_RAW[2] != RX_HDR[2]) {
    Serial.println(F("ERR bad_hdr"));
    return;
  }

  // Decode floats into BUFFER1 (channel 0 plane)
  const uint8_t *p = RX_RAW + 3;
  for (uint16_t i = 0; i < L; i++, p += 4) {
    BUFFER3[i] = read_f32_le(p);
  }

  predict_peaknet();
}
