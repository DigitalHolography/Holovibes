<p align="center">
  <img src="https://github.com/DigitalHolography/Holovibes/blob/master/Holovibes/assets/icons/Holovibes.ico"/>
</p>

## Introduction

Holovibes is designed for real-time computation of holograms from high-bitrate interferograms.
It is developed using `C++/CUDA`.

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Build and run from source (vcpkg)](#build-and-run-from-source-vcpkg)
- [Documentation](#documentation)

## Features

- Hologram computation using angular spectrum propagation and Fresnel transform.
- Time demodulation by short-time Fourier transform (STFT) and principal component analysis (PCA) algorithms.
- Graphical user interface (change parameters on the fly, keyboard shortcuts).
- Support the following cameras:
    - Ametek streaming cameras S710, S711, S991 interfaced with Euresys Coaxlink Octo & Coaxlink QSFP+ frame grabbers.
    - Adimec Quartz-2A750 interfaced with Bitflow CYT-PC2-CXP4 frame grabbers.
    - IDS imaging CMOSIS CMV 4000
    - Ximea CMOSIS XiQ and XiB
    - Hamamatsu C11440
    - Adimec Phantom S710
    - Adimec Phantom S711
- Import interferograms from `.cine` files.
- Import or export interferograms/holograms with `.holo` files.
- Modular configurations with `.json` files.
- Zone selection (zoom, signal/noise averaging, filtering).
- Many various parameters: logarithm filter, rotations, flips, ...
- Auto and manual contrast.
- Different view modes: *magnitude*, *squared magnitude*, *argument*, *unwrapped phase*.
- Batch savings (CSV).
- Batch output as raw images, controlling instruments with GPIB interface.
- Real time chart plot.

## Installation

Download the Windows installer from the [latest release](https://github.com/DigitalHolography/Holovibes/releases) and follow the installer instructions.

### Requirements

- Microsoft Windows 10 22H2, Windows 11, Windows Server 2022, or Windows Server 2025 x64.
- NVIDIA graphic card supporting CUDA 13.0, compute capability 7.5 or newer ([supported graphic cards](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)).

## Build and run from source (vcpkg)

The Windows x64 build uses CMake presets and a pinned vcpkg manifest.

### Install the build tools

1. Install Git and clone the repository into a short path to avoid Windows path-length issues:

   ```powershell
   git clone https://github.com/DigitalHolography/Holovibes.git C:\src\Holovibes
   cd C:\src\Holovibes
   ```

   If you already have a checkout, open PowerShell in its root directory instead.
2. Install Visual Studio 2022 and import [`.vsconfig`](.vsconfig) through Visual Studio Installer. It selects the Desktop development with C++ workload, MSVC 14.44, Windows SDK 10.0.26100.0, and CMake/Ninja. Keep MSVC 14.44 installed even if you also have a newer toolset; CMake 3.31 or newer is required.
3. Install CUDA Toolkit 13.0 or newer, compatible with the selected compiler, and an NVIDIA driver suitable for your toolkit and GPU. The GPU requirements above also apply when running a source build. If CUDA is installed outside its standard location, set `$env:CUDA_PATH` to the toolkit directory before building.

### Build and launch

If you previously built with Conan, check the ignored `CMakeUserPresets.json` in the repository root. If it still includes `build/generators/CMakePresets.json`, remove that obsolete include (preserving any custom presets), or rename the file as a backup if it contains only Conan-generated settings. Otherwise, CMake fails with `Could not read presets` / `File not found` before configuring the vcpkg build.

Run these commands from the repository root in PowerShell:

```powershell
.\dev.cmd build
.\dev.cmd run
```

`build` initializes the compiler environment, downloads and bootstraps vcpkg at the revision pinned in [`vcpkg.json`](vcpkg.json), restores dependencies, and builds the default `windows-dev` preset (RelWithDebInfo). You do not need to install vcpkg, Qt, or OpenCV separately. The first build can take a long time because it compiles dependencies; later builds reuse installed packages and the local binary cache.

`run` launches the existing build from `out/build/windows-dev/Holovibes.exe` with the build directory as its working directory. Run `build` again after changing source files. The wrapper works from a normal terminal and uses a process-local PowerShell execution policy.

The development preset disables camera plugins and supports recorded-file processing. To build and launch an optimized Release with available camera plugins:

```powershell
.\dev.cmd build -Preset windows-release
.\dev.cmd run -Preset windows-release
```

Camera plugins may require vendor SDKs and drivers; see [camera setup](docs/DEVELOPMENT.md#cameras-and-deployment). Use `-Preset windows-debug` on both commands for a Debug build.

### Tests and installer

| Command | Purpose |
| --- | --- |
| `.\dev.cmd dependencies` | Build and run dependency smoke tests without CUDA; does not build the application. |
| `.\dev.cmd test` | Build and run the C++ and CLI tests; GPU tests require working CUDA hardware. |
| `.\dev.cmd package` | Build Release and generate a Windows installer; requires NSIS installed. |

For direct CMake commands, build options, and binary cache configuration, see the [developer setup guide](docs/DEVELOPMENT.md).

## Documentation

- [Build from source and developer setup](docs/DEVELOPMENT.md)

- [GitHub Wiki](https://github.com/DigitalHolography/Holovibes/wiki)
- [Website](https://holovibes.com/)
- [How it works](https://docs.google.com/document/d/1H8BcAD9Gsdpc1Rs8rBjZxRaCEdW1teBxsvuC9opWElw/edit?usp=sharing)
