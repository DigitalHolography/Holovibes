# Holovibes

Holovibes is designed for real-time computation of holograms from high-bitrate interferograms.
It is developed using `C++/CUDA`.

## Features

- Hologram computation using angular spectrum propagation and Fresnel transform.
- Time demodulation by short-time Fourier transform (STFT) and principal component analysis (PCA) algorithms
- Graphical user interface (change parameters on the fly, keyboard shortcuts)
- Support the following cameras:
    - Ametek streaming cameras S710, S711, S991 interfaced with Euresys Coaxlink Octo & Coaxlink QSFP+ frame grabbers
    - Adimec Quartz-2A750 interfaced with Bitflow CYT-PC2-CXP4 frame grabbers
    - IDS imaging CMOSIS CMV 4000
    - Ximea CMOSIS XiQ and XiB
    - Hamamatsu C11440
    - Adimec Phantom S710
    - Adimec Phantom S711
- Import interferograms from `.cine` files
- Import or export interferograms/holograms with `.holo` files
- Modular configurations with `.json` files
- Zone selection (zoom, signal/noise averaging, filtering)
- Many various parameters: logarithm filter, rotations, flips, ...
- Auto and manual contrast
- Different view modes: *magnitude*, *squared magnitude*, *argument*, *unwrapped phase*
- Batch savings (CSV)
- Batch output as raw images, controlling instruments with GPIB interface
- Real time chart plot

## Requirements

- Microsoft Windows 10 pro x64 (or later)
- NVIDIA graphic card supporting CUDA 12

## Documention

- website : https://holovibes.com/
- Documentation : https://docs.google.com/document/d/1lsb1NurdSkN1GNGuATdMJ4Oubo48MSvrl203TqHH6Bs
