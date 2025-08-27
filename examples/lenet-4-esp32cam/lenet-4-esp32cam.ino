#include "esp_log.h"
#include <HardwareSerial.h>
#include <FFat.h>

#include "noodle.h"


const int IMG_SIZE = 28 * 28;
byte GRID[IMG_SIZE];
float BUFFER1[IMG_SIZE];

void setup() {
  esp_log_level_set("*", ESP_LOG_NONE);  
  
  Serial1.begin(9600, SERIAL_8N1, 3, 1); // RX=3, TX=1
  
  //while (!noodle_sd_init(14, 15, 2)) { // 1-bit pins for ESP32-CAM (AI-Thinker)
  while (!noodle_sd_init()) { // 4-bit
    delay(500);
    Serial1.print(".");
  }
  Serial1.println("SD card ok!");
}

int summarize(float et, float * buf) {
  float *y = buf;  // reuse global memory

  // Find the largest class
  int label = 0;
  float max_y = 0;

  Serial1.print("-------------------------------------\n");
  for (int i = 0; i < 10; i++) {
    Serial1.print(i);
    Serial1.print(" : ");
    Serial1.println(y[i], 2);

    if (y[i] > max_y) {
      max_y = y[i];
      label = i;
    }
  }

  Serial1.print("-------------------------------------\nPrediction: ");
  Serial1.print(label);
  Serial1.print(", in ");
  Serial1.print(et, 1);
  Serial1.println(" seconds\n");
  
  return label;
}

void predict() {
    noodle_grid_to_file(GRID, "/i01aa.txt", 28);

    unsigned long st = micros();  // timer starts
    uint16_t V;

    Serial1.println(F("Conv layer 1 ..."));
    V = noodle_conv(GRID, BUFFER1, 1, 6, "/i01xx.txt", "/o01xx.txt", "/w01xxyy.txt", "/w02.txt", 28, 2, 5, 1, 2, 2, NULL);
    
    Serial1.println(F("Conv layer 2 ..."));
    V = noodle_conv(GRID, BUFFER1, 6, 16, "/o01xx.txt", "/o02xx.txt", "/w03xxyy.txt", "/w04.txt", V, 0, 5, 1, 2, 2, NULL);
    
    Serial1.println(F("Flattening ..."));
    V = noodle_flat("/o02xx.txt", BUFFER1, V, 16);
    
    Serial1.println(F("NN layer 1 ..."));
    V = noodle_fcn(BUFFER1, V, 120, "/o03xx.txt", "/w05.txt", "/w06.txt", true, NULL);
    
    Serial1.println(F("NN layer 2 ..."));
    V = noodle_fcn("/o03xx.txt", V, 10, BUFFER1, "/w07.txt", "/w08.txt", false, NULL);

    float et = (float)(micros() - st) * 1e-6;  // timer stops

    int res = summarize(et, BUFFER1);
}

void loop() {
  static int index = 0;

  while (Serial1.available()) {
    GRID[index++] = Serial1.read();

    if (index == IMG_SIZE) {
      predict();
      Serial1.print("*"); // Done!
      index = 0; // reset for next frame
    }
  }
}

