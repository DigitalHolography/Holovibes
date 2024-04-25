## Developers

### Requirements to build Holovibes

#### Install compilers dependencies

You will need to install manually several things to build Holovibes:
* [CMake 3.27.2-msvc1 win32](https://github.com/Kitware/CMake/releases/tag/v3.27.2)
* [CUDA 12.0](https://developer.nvidia.com/cuda-downloads)
* [Python 3.7+](https://www.python.org/)
* A windows compiler, either MSVC or ClangCL (See below for further details)

#### Build cameras grabbers

Some cameras needs there own libraries to be used:
* [BitFlow SDK 6.50](http://www.bitflow.com/downloads/bfsdk65.zip)
* [Euresys EGrabber for Coaxlink](https://euresys.com/en/Support/Download-area)

#### Install a compiler

Please specify the compiler you have choosen with an option when running our dev tool documented later in the file.

##### Install MSVC (Visual Studio 2019)

Install [Visual Studio 2019](https://visualstudio.microsoft.com/fr/)

The minimum requirements in _Components_ (installable from Visual Studio installer) are the following:
* Desktop applications in C++

The minimum requirements in _Individual Components_ (installable from Visual Studio installer) are the following:
* C++ AddressSanitizer
* MSVC vXXX - VS XXXX C++ x64/x86 build tools (Latest)
* MSVC vXXX - VS XXXX C++ Spectre-mitigated Libs (Latest)

##### (NOT SUPPORTED) Install Clang (With LLVM) 

Install [LLVM](https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/LLVM-13.0.0-win64.exe)
Make sure to choose the option to put LLVm in the PATH

##### Install C++ libraries dependencies

Holovibes is using `conan` as package manager for it's C++ dependencies:
```sh
$ python -m pip install -r build/requirements.txt
$ ./dev.py conan
```

Conan will download all the dependencies missing in your system, sometimes it will build some itself if those are not compatible with your system (it can take at most one hour if you are building everything).

### Build Holovibes

Usually our dev tool will do everything for you:
Use `./dev.py` to build using our dev tool (cf. Dev tool documentation)

Alternatively, you can build from the command line directly calling conan:
`conan build . -if bin/<generator> -bf bin/<generator> -sf .`

### Test suite

#### Integration tests

The framework used here is [PyTest](https://github.com/pytest-dev/pytest)
For now integration testing is done using the CLI mode of Holovibes
and combining an input holo file, an optional configuration file and an optional file containing cli arguments.
The 3 files are passed as parameters to the CLI to create an output file which is then compared to a reference file.
The execution time is also printed, and you can compare it with the reference execution time. This can't be considered as benchmarks, and only indicates if a major change has slown down / accelerated the programm. 

##### How to add tests

An auto-discover function is already implemented.
Just create a folder in the `tests/data/` folder. This is the name of your test.
You shall put 1 to 4 files in the folder:
* an optional `input.holo` file as input (if you provide no input, the default one in the input folder will be chosen. You can provide different default inputs by making the name of the input match a part of your test name)
* a `ref.holo` file as intended output
* an optional `holovibes.json` config file for the parameters
* an oprional `cli_argument.json` containing the command line arguments

#### Build reference

From the root, run 
```sh
Holovobes>$ ./dev.py build_ref [test_name]
```
If no test_name is given, all refs will be built. It will generate a *ref.holo* or a *error.txt* and a *ref_time.txt* for each test.
**CAREFULL** : this command should be runned only for newly-added tests for which the behavior has been verified and validated by hand, or when the software's output has improved, and **if you know for sure that it's stable**

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

This will generate a *last_generated_output.holo*, or an *output_error.txt*

###### Launching only one test

To test only specific tests, you can run use

```sh
$ ./dev.py build pytest --specific-tests=test1,test2
```

It you can give it a comma separated list of folders you want to test.

#### Unit tests (NOT SUPPORTED)

Unit tests are build with [GTest](https://github.com/google/googletest)

##### Usage

Build the project in debug mode and run:
```sh
$ ./dev.py build ctest
```

### Developper Tool

Since building this project is such a hassle, we have created an unified building script to build, run and test Holovibes.

The script works like a makefile using goals to run. There are:
* conan: install the c++ packages needed by the project and put all the cmake files to find the packages in the build dir
* cmake: for cmake configure step and reconfigure step (will run conan if no build dir is found)
* build: using the generator chosen during previous step (will run cmake if no build dir is found)
* run: running the last generated executable
* ctest: running unit tests from GTest using ctest
* pytest: running integration tests using pytest
* build_ref: build the reference outputs for integration tests (run it only if the software's output improved, and **if you know for sure that it's stable**.)
* clean: remove the buid folder and the generated outputs of the integration tests

Futhermore, there is several options to manipulate the tool:
```sh
$ ./dev.py --help
usage: dev.py [-h] [-b {Release,release,R,r,Debug,debug,D,d}] [-g {Ninja,ninja,N,n,NMake,nmake,NM,nm,Make,make,M,m}]
              [-t {clang-cl,ClangCL,clangcl,Clang-cl,Clang-CL,cl,CL,MSVC,msvc}] [-p P] [-i I] [-v]

Holovibes Dev Tool (only runnable from project root)

optional arguments:
  -h, --help            show this help message and exit
  -v                    Activate verbose mode

Build Arguments:
  -b {Release,release,R,r,Debug,debug,D,d}
                        Choose between Release mode and Debug mode (Default: Debug)
Build environment:
  -p P                  Path to find the VS Developer Prompt to use to build (Default: auto-find)
  -i I                  Path used by cmake to store compiled objects and exe (Default: bin/<generator>/)
```

### Misc

#### Add an element to Front with __QtCreator__

* On _QtCreator_, load the file `mainwindow.ui`
* Add the wanted widget with a drag and drop and change its name in the right column.
* Add a slot at the end of the file `mainwindow.ui` opened in edit mode with the prototype of the function.
* Add a new slot in _QtCreator_ cliking on the green '+' on the middle.
* Fill the 4 columns with the name of your function in the last column.
* Add the prototype in `MainWindow.hh`
* Implement the function in `MainWindow.cc`

If you want to use a custom widget, you can change its class in the `mainwindow.ui` directly if it doesn't work in _QtCreator_


#### Logging

##### Reading

Logs are as follows:

```
[${log_level}] [${timestamp}] [${Thread ID}] ${logger_name} >> ${message}
```

Check logs are as follows:
```
[${log_level}] [${timestamp}] [${Thread ID}] ${logger_name} >> ${filename}:${line} ${message}
```

##### Usage

We have 5 levels of log:
* Trace (LOG_TRACE)
* Debug (LOG_DEBUG)
* Infos (LOG_INFO)
* Warnings (LOG_WARN)
* Errors (LOG_ERROR)

We have 7 loggers:
* main
* setup
* cuda
* information_worker
* record_worker
* compute_worker
* frame_read_worker

They log on std::cerr and a log file in appdata

```cpp
LOG_ERROR(logger_name, formated_string [,args]);
```

##### Assertions

Assertions are under the same banners as the logs, but here you should use the CHECK macro function as follows:
```cpp
CHECK(condition, formated_string [,args]);
```

#### Known issues


2021-10-04: If you encounter the issue `clang_rt.asan_dbg_dynamic-x86_64.dll: cannot open shared object file: No such file or directory`. You have to find the file and put it in your PATH or copy it into the build directory for it to work

2021-12-03: If `./dev.py` tells you that it cannot find conan or cmake. Please check if it's in your PATH.

2021-12-05: If conan tells you that the `XXX/system` package needs to be installed. Please install the package `XXX` manually on your system

2021-12-22: For some reason if you put a real function name like `OutputHoloFile::export_compute_settings()` in a log statement to be printed the program may segfault with an 0x5 SEH exception.
