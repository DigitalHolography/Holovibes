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

#### Visual Studio dependencies

The minimum requirements in _Individual Components_ (installable from Visual Studio installer) are the following:
* C++ CMake tools for Windows
* MSVC vXXX - VS XXXX C++ x64/x86 build tools (Latest)
* MSVC vXXX - VS XXXX C++ Spectre-mitigated Libs (Latest)

#### Environment variables

Make sure `CUDA`, `Qt`, `BitFlow` and `OpenCV` have been added to your path. *Note: it is recommended to put Qt above every other paths to avoid conflicts when loading Qt5 DLLs.*

Other variables:
* `OpenCV_DIR`: Fill with OpenCV location (Usually: `C:\opencv\build`)
* `CUDA_PATH`: Fill with Cuda and NVCC location

Do not forget to restart Visual Studio Code or your terminal before compiling again.

### Compilation

Use `./dev.py cmake build` to build using our dev tool (cf. Dev tool documentation)

By default *Ninja* is used but you can rely on other build systems (*Visual Studio 14*, *Visual Studio 15*, *Visual Studio 16* or *NMake Makefiles*) with `./build [generator]`.

Alternatively, you can build from the command line (not recommended):
* **Visual Studio**: `cmake -G "Visual Studio 14/15/16" -B build -S . -A x64 && cmake --build build --config Debug/Release`
* **Ninja**: `cmd.exe /c call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" && cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Debug/Release -DCMAKE_VERBOSE_MAKEFILE=ON && cmake --build build`

Note: After changing an element of the front or to change between release/debug mode, please delete your build folder and recompile.

### Add an element to Front with __QtCreator__

* On _QtCreator_, load the file `mainwindow.ui`
* Add the wanted widget with a drag and drop and change its name in the right collumn.
* Add a slot at the end of the file `mainwindow.ui` opened in edit mode with the prototype of the function.
* Add a new slot in _QtCreator_ cliking on the green '+' on the middle.
* Fill the 4 collumns with the name of your function in the last collumn.
* Add the prototype in `MainWindow.hh`
* Implement the function in `MainWindow.cc`


#### Known issues

2021-05-18: The project may not compile with the latest version of MSVC. To downgrade to a previous version (14.29 and 14.27 works) select the appropriate components in the `Visual Studio installer -> modify -> individual components` menu. The latest version is still required to install msvc build tools. To make MSVC use the version of your choice by default, change the file `vcvars64.bat` located in `C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build` from ``@call "%~dp0vcvarsall.bat" x64 %*`` to ``@call "%~dp0vcvarsall.bat" x64 %* -vcvars_ver=14.27``

### Test suite

#### Integration tests

pytest is used for integration testing. For now integration testing is done using the CLI
and combining an input holo file and a optional configuration file.
The 2 files are passed as parameters to the CLI to create an output file which is then compared
to a reference file.

##### How to add tests

An auto-discover function is already implemented.
Just create a folder in the `tests/data/` folder. This is the name of your test.
You shall put 2 or 3 files in the folder:
* a `input.holo` file as input
* a `ref.holo` file as intended output
* an optional `holovibes.ini` config file for the parameters

##### Usage

Just build the project either in Release or Debug mode, 
The tool used to run these tests is `pytest`. Just run this from the root of the project
```sh
$ python -m pytest -v
```

or using our dev tool:
```sh
$ ./dev.py build pytest
```

#### Unit tests

##### Installation

1. Download [GTest 1.8.1](https://github.com/google/googletest/releases/tag/release-1.8.1) and extract it to `C:/gtest`.
2. Open `googletest/msvc/2010/gtest-md.sln` with Visual Studio and build the solution in **x64** and in **Debug** mode.

##### Usage

Build the project in debug mode and run:
```sh
$ ./dev.py build ctest
```

### Dev Tool

Since building this project is such a hassle, we have created an unified building script to build, run and test Holovibes.

The script works like a makefile using goals to run. There are 5 of them:
* cmake: for cmake configure step and reconfigure step
* build: using the generator chosen during previous step
* run: running the last generated executable
* ctest: running unit tests from GTest using ctest
* pytest: running integration tests using pytest

Futhermore, there is several options to manipulate the tool:
* Build Mode:
    Choose between Release mode and Debug mode (Default: Debug)

    -b {Release,release,R,r,Debug,debug,D,d}

* Build Generator:
    Choose between NMake, Visual Studio and Ninja (Default: Ninja)

    -g {Ninja,ninja,N,n,NMake,nmake,NM,nm,Visual Studio 14,Visual Studio 15,Visual Studio 16}

* Build Environment:
    -e E                  Path to find the VS Developer Prompt to use to build
                          (Default: auto-find)
    -p P                  Path used by cmake to store compiled objects and exe
                          (Default: build/%generator%/)

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