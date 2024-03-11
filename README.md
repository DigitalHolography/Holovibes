# Holovibes

Holovibes is a software, written in `C++/CUDA`, to generate holograms from interferograms in real time.

## Features

- Hologram computation using the FFT1/FFT2/STFT/PCA algorithms
- Graphical user interface (change parameters on the fly, keyboard shortcuts)
- Support the following cameras:
    - IDS imaging CMOSIS CMV 4000
    - Ximea CMOSIS XiQ and XiB
    - Adimec Quartz-2A750 (requires [BitFlow](http://www.bitflow.com/downloads/bfsdk640.zip))
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
- Image stabilization
- Image coloring
- Image averaging in the three axis
- Lens displaying

## Requirements

- Microsoft Windows 7 x64 (or later)
- NVIDIA graphic card supporting CUDA 12.0 ([supported graphic cards](https://en.wikipedia.org/wiki/CUDA#GPUs_supported))

## Documentation

- Holovibes GitHub : https://github.com/DigitalHolography/Holovibes
- Holovibes website : https://holovibes.com/
- How Holovibes works : https://docs.google.com/document/d/1H8BcAD9Gsdpc1Rs8rBjZxRaCEdW1teBxsvuC9opWElw/edit?usp=sharing
- Build requirements, testsuite usage, logging : [here](DEVELOPERS.md)
- Setup tutorial : [here](SETUP.md)
- Contribution practices : [here](CONTRIBUTING.md)
