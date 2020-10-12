## Developers

### Dependencies

* [Visual Studio 2017 / 2019](https://visualstudio.microsoft.com/fr/) (with `Desktop applications in C++`)
* [CUDA 11.1](https://developer.nvidia.com/cuda-downloads)
* [Qt 5.9](https://download.qt.io/archive/qt/5.9/)
* [Bitflow SDK 6.30](http://www.bitflow.com/downloads/bfsdk630.zip)
* [CMake 3.16.1](https://github.com/Kitware/CMake/releases/tag/v3.16.1) (serial code 2944-8538-8655-8474)
* [Boost 1.71.0](https://boost.teeks99.com/bin/1.71.0/) (if building with CMake ; fetched from NuGet in VS)

#### Environment variables

Make sure the following environment variables are set:
    * `QTDIR` pointing to something like `C:\Qt\Qt5.x.x\5.x.x\msvc2017_64\`

* Make sure your path contains:
    * $(CUDA_PATH_V11_1)\bin
    * $(CUDA_PATH_V11_1)\libnvvp
    * $(QTDIR)\bin

#### Common issues

* Bitflow SDK issues
    * When installing you MUST write the serial code or you will be missing critical dlls.

* QT issues
    * After modifying your path, if Holovibes cannot find the Qt platform "windows", redownload Qt.
    * Verify that it builds with the correct dll. If not, your path contain something wrong.
    * If asked to set Qt5_DIR var (by Cmake), only set QTDIR (as asked), delete holovibes folder, reset Cmake cache (in cmake gui) and clone again.

* Boost issues
    * If you get a problem linked to "boost program_options" you can try running this command : (name might not be b2.exe but bjam.exe)
    * `b2.exe install --prefix="YOUR_PATH\boost\boost_1_7X_X" --with-system --with-date_time --with-random link=static runtime-link=shared threading=multi`
    * This command will install program_options which may not be included with boost itself.

* General issues
    * Reload the terminal, visual studio or even your computer.
    * Clean cmake cache (cmake gui).
    * Check that you have exactly the same versions as mention.

### Compilation

* Have cmake 3.16.1 installed
* Run `./build.py [OPTIONS]` in the main directory
    * `./build.py` or `./build.py D`: debug mode
    * `./build.py R`: release mode
    * `./build.py [Gen]`: uses `Gen` as a generator (Ninja by default)
* Run `./run.py [OPTIONS]` to run the code
    * `./run.py` or `./run.py D`: run in debug mode
    * `./run.py R`: run in release mode
* Alternatively you can use Visual Studio 2019's CMake feature by selecting the right CMake version
* You could also build from the command line
    * `cmake -G "Visual Studio 14/15/16" -B build -S . -A x64 && cmake --build build --config Debug/Release` to build with Visual Studio
    * `cmd.exe /c call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" && cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Debug/Release -DCMAKE_VERBOSE_MAKEFILE=ON && cmake --build build` to build with Ninja (it is faster)

### Unit tests

#### Install GTest

1. Download [GTest 1.8.1](https://github.com/google/googletest/releases/tag/release-1.8.1) and extract it in `C:/gtest`.
2. Open `googletest/msvc/2010/gtest-md.sln` with Visual Studio and build the solution in **x64** and in **Debug** mode.

#### Usage

Build the project in debug mode and run all unit tests with `./run_tests.py`.

### Making a new release

* Pull / merge / rebase all the features that need to be included on master
* Change the version number in `compute_descriptor.hh`
* Make a clean in build in release mode `rm -rf build && ./build.py R`
* Make sure everything works as intended
* Change the version number in `README.md` and write a short changelog
* Change the version number in `setupCreator.iss`
* Add any missing dll in `setupCreator.iss` (just like the lines 115 to 135)
* Commit your changes `git add -u && git commit -m "vX.X"`
* Tag your commit `git tag -a "vX.X" -m "vX.X" && git push --tags origin master`
* Run the `setupCreator.iss` script using ino setup
* Use the installer located in the `Output` directory and make sure everything still works as intended
* Upload the installer on `ftp.espci.fr` in the `incoming/Atlan/holovibes/installer/Holovibes_vX.X.X`

### Adding a camera

* Right click on "Camera Libbraries"
* Add -> new project -> empty project
* right click on your project -> properties -> change '.exe' to '.dll'
* View -> property Manager
* For both "Debug | x64" and "Release | x64":
    * Add -> Add existing property sheet -> "PropertySheet.props" and "CameraDLL.props"
* Make sure "CAMERA_EXPORTS" is set (properties -> C/C++ -> preprocessor -> Preprocessor definitions)
* right click on Holovibes -> Build Events -> Post-Build Events -> add the copy of your SDk dll and your ini file
* Do not forget updating setupCreator.iss to copy your newly created Dll and .ini file