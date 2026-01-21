<p align="center">
  <img src="./docs/attachments/noodle.png" alt="Description" width="100"> 
</p>
<h1 align="center">Noodle 🍜≈🧠</h1>
<h4 align="center">Lightweight neural network inference engine with file streaming support</h4>

---


Noodle provides primitive modular functions for **convolution layer, dense layer, pooling and activations**. It allows streaming the intermediate activations, weights, and biases from/to **SD/FFat/SD_MMC** filesystems to overcome RAM limitations. During the development, we typically test Noodle with low-tier MCUs, such as: Arduino Uno R3, UNO R4, Mega256, and some ESP32 variants.


## Documentation

Please check this GitHub Pages: https://auralius.github.io/noodle/

## Persistent identifier

[DOI: 10.5281/zenodo.16239227](https://doi.org/10.5281/zenodo.16239228)

## Special Notes

* Training is done with Keras with PyTorch back-end.  

* [model_exporter.py](./model_exporter.py) is used to export the weights/biases into files.  

* Although we still use the Arduino Framework, development is done wit Visual Code and PlatformIO.  


## Authors

- Auralius Manurung — Universitas Telkom, Bandung -- auralius.manurung@ieee.org  
- Lisa Kristiana — ITENAS, Bandung


## Copyright and license

Code released under the MIT License. Docs released under Creative Commons.



