This is a demonstration for LeNet-5 for single digit classifier.

There are 3 environments:

* For ESP32 development kit that has abuilt in touch display, known as Cheap Yellow Display: `[env:cyd_esp32]`  

* For plain ESP32: `[env:esp32]`

* For Raspberry Pi Pico: `[env: rpi_pico]`

There are two schemes:

* CNN in file & FCN in variable, for ESP32, requires FFAT-paritioned flash

* CN in variable & FCN in variable, for ESP32 and Raspberry Pi Pico
