# Tutorial

- [1. Introduction](#1-introduction)
- [2. Setup and installation](#2-setup-and-installation)
   - [2.1 Requirements](#21-requirements)
   - [2.2 Initial setup](#22-initial-setup)
   - [2.3 Building the project](#23-building-the-project)
- [3. Project structure](#3-project-structure)
   - [3.1 File structure](#31-file-structure)
   - [3.2 Code interaction: UI and API](#32-code-interaction-ui-and-api)
- [4. Coding Standards](#4-coding-standards)
   - [4.1 Code format and style](#41-code-format-and-style)
   - [4.2 Git](#42-git)
- [5 Developing New Features](#5-developing-new-features)
   - [5.1 Creating a New Feature Branch](#51-creating-a-new-feature-branch)
   - [5.2 Creating Pull Requests](#52-creating-pull-requests)
   - [5.3 Creating a Release](#53-creating-a-release)
- [6. Advanced Topics](#6-advanced-topics)
   - [6.1 Benchmarking](#61-benchmarking)
   - [6.2 Logging](#62-logging)
   - [6.3 Assertions](#63-assertions)
   - [6.4 Adding elements to UI](#64-adding-elements-to-front-with-qtcreator)
   - [6.5 Updating the UI from the API](#65-updating-the-ui-from-the-api)
- [7. Troubleshooting](#7-troubleshooting)
- [8. Documentation](#8-documentation)
   - [8.1 Generate documentation](#81-generate-documentation)
   - [8.2 Write documentation](#82-write-documentation)
- [9. Tests](#9-test)
   - [9.1 Add a test](#91-add-a-test)
   - [9.2 Build reference outputs](#92-build-reference-outputs)
   - [9.3 Run tests](#93-run-tests)
- [10. Tools](TUTORIAL.md#10-tools)
   - [10.1 Dev.py](#101-devpy-tools)
   - [10.2 Holo file inspector](#102-holo-file-inspector)
   - [10.3 Benchmark viewer](#103-benchmark-viewer)
   - [10.4 Python scripts](TUTORIAL.md#10-tools)
      - [10.4.1 Convert holo](TUTORIAL.md#convert_holopy)
      - [10.4.2 Add missing compute settings](TUTORIAL.md#add_missing_compute_settingspy)

# 1. Introduction

Holovibes is designed for real-time computation of holograms from high-bitrate interferograms. It is developed using `C++/CUDA` and supports various cameras and file formats. This tutorial will guide you through the process of setting up your development environment, understanding the codebase, and contributing effectively to the project.

# 2. Setup and installation

## 2.1 Requirements

To develop for Holovibes, ensure you have the following software installed:

- GIT
- CMake
- Visual Studio 2022 with C++ Desktop Development
- CUDA 12.6
- Python 3.8.10
- NSIS
- Conan 2.7.0

## 2.2 Initial setup

1. **Install GIT** from [here](https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe).
2. **Install Visual Studio 2022** from [here](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false).
   - The minimum requirements in _Components_ (installable from Visual Studio installer) are the following:
      - Desktop applications in C++

   - The minimum requirements in _Individual Components_ (installable from Visual Studio installer) are the following:
      - C++ AddressSanitizer
      - MSVC vXXX - VS XXXX C++ x64/x86 build tools (Latest)
      - MSVC vXXX - VS XXXX C++ Spectre-mitigated Libs (Latest)
3. **Install CUDA 12.6** from [here](https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda_12.6.1_560.94_windows.exe)

4. **Add Paths to PATH Environment Variables**:
   - `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`
   - **Depending on your VS version (commonly *Community*):**
      - `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64`
      - `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x86`
      **OR**
      - `C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64`
      - `C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x86`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
5. **Reboot your PC**.
6. **Install Python 3.8.10** from [here](https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe). Ensure to tick `Add Python 3.8 to PATH`.
7. **Install NSIS** from [here](https://sourceforge.net/projects/nsis/files/NSIS%203/3.09/nsis-3.09-setup.exe/download?use_mirror=netcologne&download=).
8. **Install Conan 2.7.0**:
   ```sh
   $ pip install conan && conan profile detect --force
   ```

>Bitflow frames grabber and euresys frames grabber SDKs must be installed on the computer when making a release executable or it will not contains the dll required to load the camera.
> - [BitFlow SDK 6.6](http://www.bitflow.com/downloads/bfsdk66.zip)
> - [Euresys EGrabber for Coaxlink](https://euresys.com/en/Support/Download-area)

## 2.3 Building the project

All commands (except cpack) must be run at the root of the project:

1. **Install Dependencies (once)**:
   ```sh
   $ conan install . --build missing --settings compiler.cppstd=20 --settings build_type=Release
   ```
   OR
   ```sh
   $ ./dev.py install
   ```

2. **Configure the CMake Project**:
   ```sh
   $ build\generators\conanbuild.bat && cmake --preset conan-release
   ```
   OR
   ```sh
   $ ./dev.py cmake
   ```

3. **Build the Project**:
   ```sh
   $ build\generators\conanbuild.bat && cmake --build --preset conan-release
   ```
   OR
   ```sh
   $ ./dev.py build
   ```

4. **Run the Project**:
   ```sh
   $ ./dev.py run
   ```

5. **Create an Installer (you must be in the build/bin folder)**:
   ```sh
   $ cpack
   ```

# 3. Project Structure

## 3.1 File structure

- **build/**: Contains build-related files.
- **Camera/**: Contains interface for using cameras. And specific implementation and configs file for each camera.
- **docs/**: Contains documentation files (doxygen, diagram).
- **Holovibes/convolution_kernels**: txt files containing numbers for the different kernel (sobel, gaussian blur).
- **Holovibes/includes/**: Contains header files.
- **Holobives/sources/**: Contains source code of the project.
- **Preset**: Contains different parameters preset for holovibes stored as `.json` files.
- **tests/**: Contains test cases.
- **CHANGELOG.md**: Keep track of all user related changes (UI, features) for each version.

## 3.2 Code interaction: UI and API

The UI calls API functions and listens to a notifier using a subscriber/observer pattern to synchronize changes between different UIs.

- **UI Calls API Functions**: The UI elements trigger actions and request data by calling corresponding API functions.
- **Notifier and Subscriber**: The notifier mechanism is used to alert subscribers (UI components) about changes. This ensures that any update in the data or state is reflected across all UI components consistently. (Refer to [Updating the UI from the API](#63-updating-the-ui-from-the-api))

# 4. Coding standards

## 4.1 Code format and style

On Visual Studio Code you can install the extension `Clang-format` and enable `Format On Save` in preferences.

Install a pre-commit hook with `pip install pre-commit` and then use `pre-commit install` at the root of the project.

Adhere to the following coding standards to maintain code consistency:

- Use `.clang-format` for formatting.
- No brackets in one line ifs
- Follow the naming conventions:
   - Classes, Structs, Unions and Enums must be named in **CamelCase**.
   - Class Members, Variables, Namespaces, Functions and Files must be named in **snake_case**.
   - Getters and setters must be named `get_${var}` and `set_${var}` respectivly.
   - Class members' name must end with a `_` (ex: `frame_size_`).
- Ensure all header files have the extension `.hh`, source files `.cc`, and templated classes `.hxx`.
- Getters, Setters and inlinable functions must be written in `*.hh` files.
- Use Doxygen for comments (ex: `/* ! \brief This is an example */`) and document your code.

To clang format the entiere project:
```sh
find . \( -name "*.hh" -o -name "*.hxx" -o -name "*.cc" -o -name "*.cuh" -o -name "*.cuhxx" -o -name "*.cu" \) -exec clang-format.exe -i {} \;
```

## 4.2 Git

- Branch `master` is only used for stable releases.
- Never push trash or generated files.
- Always work on separate branches when developing a new feature or fixing a bug.
- Use pull request when you want to merge your branch into dev.
- Use `git pull --rebase` if possible to avoids useless merge commits.
- Use the [React](https://github.com/ubilabs/react-geosuggest/blob/main/CONVENTIONS.md) commit convention.

# 5. Developing New Features

## 5.1 Creating a New Feature Branch

Always create a new branch from `dev` for any new feature or bug fix:

```sh
git checkout -b feature/new-feature dev
```
or
```sh
git switch -c feature/new-feature dev
```

## 5.2 Creating Pull Requests

1. **Add a changelog entry**:
   - Add a new entry in the `CHANGELOG.md` file.
   - Follow the format: `## [Unreleased]` -> `## [Version]`.
2. **Commit Your Changes**:
   Your commit message must follow the [React convention](https://github.com/ubilabs/react-geosuggest/blob/main/CONVENTIONS.md).
   ```sh
   git add .
   git commit -m "feat(scope): add new feature"
   ```
3. **Push Your Branch**:
   ```sh
   git push origin feature/new-feature
   ```
4. **Create a Pull Request**: Go to the GitHub repository and create a pull request from your feature branch to `dev`.

## 5.3 Creating a release

Before making a release make sure you have the cameras SDK.

1.  Merge all your feature branches on `dev`.
2.  Update `CHANGELOG.md`.
3.  Apply the Benchmark protocol.
4.  Make a clean build in release mode (`./dev.py clean build -b Release`).
5.  Make sure everything works as intended and run test suite (`./dev.py pytest -b Release`). (`ctest` aren't working and are not to be used for now).
6.  Make sure the git repository has no work in progress.
7.  Run the release script with `./dev.py -b Release release {bump-type}` with `bump-type` being either `major`, `minor` or `patch`
    if you want to modify either X, Y, or Z of the version number X.Y.Z.
8.  add the updated CMakeLists.txt to git.
9.  do a `git push --follow-tags`.
10.  Merge master with `dev`.

# 6. Advanced Topics

## 6.1 Benchmarking

It's a way to compare the performance of different versions of the project. It's important to run it before creating a new release.
Create the `[appdata]/Holovibe/[version]/benchmark` folder (if it doesn't exist), benchmarking csv will be saved in this folder. Then do a clean compile in debug mode and run
```sh
dev.py run --benchmark
```

To compare two benchmark files use the [Benchmark viewer](#103-benchmark-viewer)

*You are invited to improve the protocol, the benchmark informations gathering in the Information Worker and the BenchmarkViewer file.*

## 6.2 Logging

We have 5 levels of log:
- Trace (`LOG_TRACE`)
- Debug (`LOG_DEBUG`)
- Infos (`LOG_INFO`)
- Warnings (`LOG_WARN`)
- Errors (`LOG_ERROR`)

And 7 loggers that log on std::cerr and in a log file in appdata:
- main
- setup
- cuda
- information_worker
- record_worker
- compute_worker
- frame_read_worker

Usage:
```cpp
LOG_ERROR(logger_name, formated_string [,args]);
```

Format:
```
[${log_level}] [${timestamp}] [${Thread ID}] ${logger_name} >> ${message}
```

## 6.3 Assertions

Usage:
```cpp
CHECK(condition)
CHECK(condition, formated_string [,args]); // Up to 14 varags
```

Format:
```
[${log_level}] [${timestamp}] [${Thread ID}] ${logger_name} >> ${filename}:${line} ${message}
```

## 6.4 Adding Elements to Front with QtCreator

- In _QtCreator_, load a `.ui` file
- Add a widget and change its name in the right column.
- Open as plain text the `.ui` file and add in the `<slots>` element at the end of the file a slot with the prototype of the function (ex: `<slot>functionName(args)</slot>`).
- Add a new slot in _QtCreator_ cliking on the green '+' on the middle.
- Fill the 4 columns with the name of your function in the last column.
- Add the prototype in `MainWindow.hh` and implement it in `MainWindow.cc`.

If you want to use a custom widget, you can change its class in the `.ui` file directly if it doesn't work in _QtCreator_.

## 6.5 Updating the UI from the API

1. Create a new notifier in the API.
   - Choose a name that reflects the information being sent.
   - Choose a type for the information sent.
      - You must send at least one argument. If you don't need to send any information, use a boolean.
      - If you want to send multiple arguments, either use a struct or a tuple.

   ```cpp
   auto& manager = NotifierManager::get_instance();
   auto notifier = manager.get_notifier<type>("notifier_name");
   notifier->notify(data);
   ```
2. Create a new subscriber in the UI.
   - Add it in the constructor.
   ```cpp
   // In the .hh
   void on_notifier_name(const type& data);
   Subscriber<type> notifier_name_subscriber;
   // In the .cc
   Class::Constructor()
      : notifier_name_subscriber("notifier_name",
         std::bind(&Class::on_notifier_name, this, std::placeholders::_1))
   ```
   - Implement the function that will be called when the notifier is triggered.

# 7. Troubleshooting

- 2021-10-04: If you encounter the issue `clang_rt.asan_dbg_dynamic-x86_64.dll: cannot open shared object file: No such file or directory`. You have to find the file and put it in your PATH or copy it into the build directory for it to work
- 2021-12-03: If `./dev.py` tells you that it cannot find conan or cmake. Please check if it's in your PATH.
- 2021-12-05: If conan tells you that the `XXX/system` package needs to be installed. Please install the package `XXX` manually on your system
- 2021-12-22: For some reason if you put a real function name like `OutputHoloFile::export_compute_settings()` in a log statement to be printed the program may segfault with an 0x5 SEH exception.
- If the app crashes after launch, try removing the app settings from `C:\Users\[user]\AppData\Roaming\Holovibes\[version]\*`.
- If the app crashes and tells you that camera configs where not found in the AppData, you can (or):
   - Install the release matching your current version and launch it (it will setup the folder in AppData)
   - Copy the `Camera/configs` folder and paste its content (except the bfml folder) into `%APPDATA%/Holovibes/[version]/cameras_config`

# 8. Documentation

## 8.1 Generate documentation

Generate the documentation using Doxygen (you need to install it before):

```sh
$ ./dev.py doc
```

This will compile the documentation and open it in your default browser.

## 8.2 Write documentation

Follow [these recommandation](docs/Doxygen/DOCUMENTING.md)

# 9. Test

The framework used is [PyTest](https://github.com/pytest-dev/pytest). Integration testing is done using the CLI mode of Holovibes.

## 9.1 Add a test

Create a folder with the name of your test in the `tests/data/` folder. Then put:
- an optional `input.holo` (if you provide no input, the default one in the input folder will be chosen. You can provide different default inputs by making the name of the input match a part of your test name).
- a `ref.holo` file as intended output.
- an optional `holovibes.json` containing parameters.
- an oprional `cli_argument.json` containing CLI arguments.

## 9.2 Build Reference Outputs

At the root of a stable branch run:
```sh
./dev.py build_ref [test_name]
```
If no `test_name` is given, all refs will be built. It will generate a `ref.holo` or a `error.txt` and a `ref_time.txt` for each test.
**CAREFULL** : this command should only be runned in a stable branch and for newly-added tests for which the behavior has been verified and validated by hand, or when the software's output has improved, and **if you know for sure that it's stable**.

## 9.3 Run tests

Build the project in Release or Debug mode.
If you want all tests run:
```sh
$ python -m pytest -v
```
OR
```sh
$ ./dev.py build pytest
```

If you want to run one or specific tests run:
```sh
./dev.py build pytest --specific-tests=names
```
Where `names` is a comma separated list of folders

Both will generate a `last_generated_output.holo`, or an `output_error.txt`.

# 10. Tools

## 10.1 Dev.py tools

The script works like a makefile using goals to run. There are:
- conan: install the c++ packages needed by the project and put all the cmake files to find the packages in the build dir.
- cmake: for cmake configure step and reconfigure step (will run conan if no build dir is found).
- build: using the generator chosen during previous step (will run cmake if no build dir is found).
- run: run the last generated executable.
- ctest: run unit tests from GTest using ctest.
- pytest: run integration tests using pytest.
- build_ref: build the reference outputs for integration tests (run it only if the software's output improved, and **if you know for sure that it's stable**.)
- clean: remove the buid folder and the generated outputs of the integration tests

You can run `./dev.py --help` for more options.

## 10.2 Holo file inspector

Allow to view the header and extract/change parameters inside a `.holo` file. The tools can be downloaded at [each releases](https://github.com/DigitalHolography/Holovibes/releases).

The format of these files can be found [here](docs/Holo%20file%20format.md).

## 10.3 Benchmark viewer

[Benchmark viewer](https://github.com/TitouanGragnic/benchmark-viewer)

## 10.4 Holovibes Python scripts

### Requirements

*Note: It is recommended to move the `python/` folder from Holovibes installation directory (usually `C:/Program Files/Holovibes/X.X.X/`) to another place to avoid permissions issues.*

1. Have [python3](https://www.python.org/downloads/) installed
2. Install dependencies with `pip install -r requirements.txt`

### convert_holo.py

| From  | To    | Command                                         |
|-------|-------|-------------------------------------------------|
| .holo | .avi  | `python3 convert_holo.py input.holo output.avi` |
| .holo | .mp4  | `python3 convert_holo.py input.holo output.mp4` |
| .holo | .raw  | `python3 convert_holo.py input.holo output.raw` |
| .raw  | .holo | `python3 convert_holo.py input.raw output.holo` |

For .avi and .mp4 you can specify the output video FPS (by default 20) with `--fps 30`.

*Note: when creating a .holo from a .raw the program will prompt the user mandatory parameters: width, height, bytes per pixel and number of frames.*

### add_missing_compute_settings.py

Used to add/remove compute_settings from .holo and .json when changes have been made in the code.

The script must be run in the tests folder :
```sh
Holovibes/tests>$ ./add_missing_compute_settings.py [json_name] [holo_name]
```
json_name and holo_name are the .json and .holo that the script will modify. If you don't specify the names, it will modify all the **holovibes.json** and **ref.holo** recursively in the current folder and subfolders.

In order to add and remove keys, modify the directory "new_key_values" in the file. Here is an example usage:
```py
directory_path = '.'  # Change this to the directory where your JSON files are located
new_key_values = {
    ("image_rendering", "input_filter"): {"enabled": False, "type": "None"}, # Adds a key
    ("color_composite_image", "hsv", "slider_shift"): None, # Removes the key
    ("color_composite_image", "hsv", "h", "slider_shift"): {"max": 1.0,"min": 0.0}, # Adds the key
    ("color_composite_image", "hsv", "h", "frame_index", "activated"): False, # Adds the key
    ("color_composite_image", "hsv", "h", "blur"): None, # Removes the key
}
```
