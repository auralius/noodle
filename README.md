<p align="center">
  <img src="./docs/attachments/noodle.png" alt="Description" width="100"> 
</p>

<h1 align="center">Noodle 🍜≈🧠</h1>


**Noodle** is a lightweight convolutional neural network inference library designed for MCUs with **very small RAM**.  

It streams activations and weights from **SD/FFat/SD_MMC** filesystems to overcome RAM limitations. Noodle provides primitive modular functions for **convolution layer, dense layer, pooling and activations**. During the development, we typically test Noodle with Arduino UNO R3, UNO R4, and some ESP32 variants.


## Documentation

Please check this GitHub Pages: https://auralius.github.io/noodle/

## Persistent identifier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16239228.svg)](https://doi.org/10.5281/zenodo.16239228)


## Special Notes

* Training is done with Keras with PyTorch back-end.  

* [model_exporter.py](./model_exporter.py) is used to export the weights/biases into files.  

* Although we still use the Arduino Framework, development is done wit Visual Code and PlatformIO.  


## Authors

- Auralius Manurung — Universitas Telkom, Bandung -- auralius.manurung@ieee.org  
- Lisa Kristiana — ITENAS, Bandung


## Copyright and license

Code released under the MIT License. Docs released under Creative Commons.


