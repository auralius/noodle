This is a demonstration for LeNet-5 for single digit classifier.

There are three environments:

* For ESP32 development kit that has abuilt in touch display, known as Cheap Yellow Display: `[env:cyd_esp32]`  

* For plain ESP32: `[env:esp32]`

* For Raspberry Pi Pico: `[env: rpi_pico]`

There are two schemes:

* CNN in files & FCN in variables, for ESP32, this sheme requires FFAT-paritioned flash

* CNN in variables & FCN in variables, for ESP32 and Raspberry Pi Pico
