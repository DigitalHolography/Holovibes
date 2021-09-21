## Developers

### Dependencies

* [Visual Studio 2019](https://visualstudio.microsoft.com/fr/) (with `Desktop applications in C++`)
* [CMake 3.19.0-rc1 win32](https://github.com/Kitware/CMake/releases/tag/v3.19.0-rc1)
* [CUDA 11.2](https://developer.nvidia.com/cuda-downloads)
* [Qt 5.9](https://download.qt.io/archive/qt/5.9/)
* [Boost 1.71.0](https://boost.teeks99.com/bin/1.71.0/)
* [BitFlow SDK 6.50](http://www.bitflow.com/downloads/bfsdk65.zip) (serial code 2944-8538-8655-8474)
* [Euresys EGrabber for Coaxlink](https://euresys.com/en/Support/Download-area)
* [OpenCV 4.5.0](https://opencv.org/releases/)

#### Environment variables

Make sure `CUDA`, `Qt`, `BitFlow` and `OpenCV` have been added to your path. *Note: it is recommended to put Qt above every other paths to avoid conflicts when loading Qt5 DLLs.*

Other variables:
    - `OpenCV_DIR` = `C:\opencv\build`

Do not forget to restart Visual Studio Code or your terminal before compiling again.

### Compilation

After changing element of the front, changing release/debug mode, delete your build folder and recompile.

Use `./build.py` (or `./build.py R` for release mode / `./build.py P` if using Visual Studio professional) and `./run.py` (or `./run.py R` for release mode) in project folder.

By default *Ninja* is used but you can rely on other build systems (*Visual Studio 14*, *Visual Studio 15*, *Visual Studio 16* or *NMake Makefiles*) with `./build [generator]`.

Alternatively, you can build from the command line:
* **Visual Studio**: `cmake -G "Visual Studio 14/15/16" -B build -S . -A x64 && cmake --build build --config Debug/Release`
* **Ninja**: `cmd.exe /c call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" && cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Debug/Release -DCMAKE_VERBOSE_MAKEFILE=ON && cmake --build build`

### Add an element to Front with __QtCreator__

* On _QtCreator_, load the file `mainwindow.ui`
* Add the wanted widget with a drag and drop and change its name in the right collumn.
* Add a slot at the end of the file `mainwindow.ui` opened in edit mode with the prototype of the function.
* Add a new slot in _QtCreator_ cliking on the green '+' on the middle.
* Fill the 4 collumns with the name of your function in the last collumn.
* Add the prototype in `MainWindow.hh`
* Implement the function in `MainWindow.cc`


#### Known issues

2021-05-18: The project does not compile with the latest version of MSVC (14.29). To downgrade to a previous version (14.27 works) select the appropriate components in the `Visual Studio installer -> modify -> individual components` menu. The latest version is still required to install msvc build tools. To make MSVC use the version of your choice by default, change the file `vcvars64.bat` located in `C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build` from ``@call "%~dp0vcvarsall.bat" x64 %*`` to ``@call "%~dp0vcvarsall.bat" x64 %* -vcvars_ver=14.27``

### Test suite

#### Integration tests

##### Usage

Build the project in release mode and run all integration tests with `./run_integration_tests.py`.

#### Unit tests

##### Installation

1. Download [GTest 1.8.1](https://github.com/google/googletest/releases/tag/release-1.8.1) and extract it to `C:/gtest`.
2. Open `googletest/msvc/2010/gtest-md.sln` with Visual Studio and build the solution in **x64** and in **Debug** mode.

##### Usage

Build the project in debug mode and run all unit tests with `./run_unit_tests.py`.

### Misc

#### Logging

##### Reading

Logs are as follows:

```
${datetime} <${time from start}> [${Thread ID}] ${filename}:${line_in_file} ${log_level}| ${message}
```

##### Usage

We have 5 levels of log:
* Trace (LOG_TRACE)
* Debug (LOG_DEBUG)
* Infos (LOG_INFO)
* Warnings (LOG_WARN)
* Errors (LOG_ERROR)

They are usable as std:cout and any C++ Stream.
For instance, if a file named `config.json` is not found, you could write:
```cpp
LOG_ERROR << "File named config.json could not be found";
```

##### Assertions

Assertions are under the same banners as the logs, but here you should use the CHECK macro function as follows:
```cpp
CHECK(condition) << "An error occured";
```