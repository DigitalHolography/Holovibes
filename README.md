# Holovibes

Holovibes is a software, written in `C++/CUDA`, to generate holograms from interferograms in real time.

## Features

- Hologram computation using angular spectrum propagation and Fresnel transform.
- Time demodulation by short-time Fourier transform (STFT) and principal component analysis (PCA) algorithms
- Graphical user interface (change parameters on the fly, keyboard shortcuts)
- Support the following cameras:
    - Ametek streaming cameras S710, S711, S991 interfaced with Coaxlink Octo frame grabbers
    - Adimec Quartz-2A750 interfaced with Bitflow CYT-PC2-CXP4 frame grabbers
    - IDS imaging CMOSIS CMV 4000
    - Ximea CMOSIS XiQ and XiB
    - Hamamatsu C11440
- Import interferograms from `.cine` files (Ametek/VisionResearch proprietary file format for high-speed camera footage)
- Import or export interferograms/holograms with `.holo` files (http://holofile.org/)
- Modular advanced configurations with `.json` files
- Zone selection (zoom, signal/noise averaging, filtering)
- Basic post-processing options: Image filtering, rotation, flip, ...
- Auto and manual contrast
- Batch savings (CSV)
- Batch output as raw images, controlling instruments with GPIB interface
- Real time chart plot

## Requirements

- Microsoft Windows 10 pro x64 (or later)
- NVIDIA graphic card supporting CUDA 12

## Documention

- website : https://holovibes.com/
- Documentation : https://docs.google.com/document/d/1H8BcAD9Gsdpc1Rs8rBjZxRaCEdW1teBxsvuC9opWElw/edit?usp=sharing
