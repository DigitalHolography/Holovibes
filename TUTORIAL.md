
# TUTORIAL.md

Welcome to the Holovibes developer's tutorial. This document aims to provide a comprehensive guide for developers tasked with implementing new features on Holovibes. This tutorial covers setting up the development environment, understanding the project structure, contributing code, and testing your implementations.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup and Installation](#setup-and-installation)
   - [Requirements](#requirements)
   - [Initial Setup](#initial-setup)
   - [Building the Project](#building-the-project)
3. [Project Structure](#project-structure)
4. [Coding Standards](#coding-standards)
5. [Developing New Features](#developing-new-features)
   - [Creating a New Feature Branch](#creating-a-new-feature-branch)
   - [Implementing Features](#implementing-features)
   - [Testing Your Code](#testing-your-code)
   - [Creating Pull Requests](#creating-pull-requests)
   - [Creating a Release](#creating-a-release)
6. [Advanced Topics](#advanced-topics)
   - [Benchmarking](#benchmarking)
   - [Logging](#logging)
   - [Miscellaneous](#miscellaneous)
7. [Troubleshooting](#troubleshooting)
8. [Documentation](#documentation)

## Introduction

Holovibes is designed for real-time computation of holograms from high-bitrate interferograms. It is developed using `C++/CUDA` and supports various cameras and file formats. This tutorial will guide you through the process of setting up your development environment, understanding the codebase, and contributing effectively to the project.

## Setup and Installation

### Requirements

To develop for Holovibes, ensure you have the following software installed:

- GIT
- Visual Studio 2022 with C++ Desktop Development
- CUDA 12.2
- Python 3.8.10
- NSIS
- Conan

### Initial Setup

Follow these steps to set up your development environment:

1. **Install GIT**: Download and install GIT from [here](https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe).

2. **Install Visual Studio 2022**: Download and install Visual Studio 2022 from [here](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false).

3. **Add Paths to Environment Variables**:
   - `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`
   - `C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64`
   - `C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x86`

4. **Install CUDA 12.2**: Download and install CUDA 12.2 from [here](https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local).

5. **Install Python 3.8.10**: Download and install Python from [here](https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe). Ensure to tick `Add Python 3.8 to PATH`.

6. **Install NSIS**: Download and install NSIS from [here](https://sourceforge.net/projects/nsis/files/NSIS%203/3.09/nsis-3.09-setup.exe/download?use_mirror=netcologne&download=).

7. **Install Conan**:
   ```sh
   pip install conan && conan profile detect --force
   ```

### Building the Project

To build Holovibes, follow these steps:

1. **Install Dependencies**:
   ```sh
   Holovibes>$ conan install . --build missing --settings compiler.cppstd=20 --settings build_type=Release
   ```
   OR
   ```sh
   Holovibes>$ ./dev.py install
   ```

2. **Configure the CMake Project**:
   ```sh
   Holovibes>$ build\generators\conanbuild.bat && cmake --preset conan-release
   ```
   OR
   ```sh
   Holovibes>$ ./dev.py cmake
   ```

3. **Build the Project**:
   ```sh
   Holovibes>$ build\generators\conanbuild.bat && cmake --build --preset conan-release
   ```
   OR
   ```sh
   Holovibes>$ ./dev.py build
   ```

4. **Run the Project**:
   ```sh
   Holovibes>$ ./dev.py run
   ```

5. **Create an Installer**:
   ```sh
   Holovibes\build\bin>$ cpack
   ```

## Project Structure

Understanding the project structure is crucial for effective development. Here is a high-level overview:

- **Holobives/sources/**: Contains the source code of the project.
- **Holovibes/includes/**: Contains the header files.
- **tests/**: Contains the test cases.
- **build/**: Contains build-related files.
- **docs/**: Contains documentation files.

### Code Interaction: UI and API

The interaction between the UI and the API in Holovibes follows a structured approach where the UI calls API functions and listens to a notifier using a subscriber pattern to synchronize changes between different UIs.

- **UI Calls API Functions**: The UI elements trigger actions and request data by calling corresponding API functions.
- **Notifier and Subscriber**: The notifier mechanism is used to alert subscribers (UI components) about changes. This ensures that any update in the data or state is reflected across all UI components consistently. (Refer to [Updating the UI from the API](#updating-the-ui-from-the-api))

## Coding Standards

Adhere to the following coding standards to maintain code consistency:

- Use `.clang-format` for formatting.
- Follow the naming conventions:
  - Classes, Structs, Unions, Enums: CamelCase
  - Variables, Functions, Files: snake_case
- Use Doxygen for comments.
- Ensure all header files have the extension `.hh`, source files `.cc`, and templated classes `.hxx`.

Refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for detailed coding standards.

## Developing New Features

### Creating a New Feature Branch

Always create a new feature branch from `dev` for any new feature or bug fix:

```sh
git checkout -b feature/new-feature dev
```
or
```sh
git switch -c feature/new-feature dev
```

### Implementing Features

1. **Add New Code**: Implement your feature in the appropriate files.
2. **Follow Coding Standards**: Ensure your code adheres to the project's coding standards.
3. **Update Documentation**: Update relevant documentation to reflect your changes.

### Testing Your Code

1. **Integration Tests**: Add integration tests in the `tests/data/` folder.
2. **Build Reference Outputs**:

   On a stable branch, you can build reference outputs for the tests. Don't run this command on a branch that is not stable as it might introduce errors to the reference outputs.

   ```sh
   Holovibes>$ ./dev.py build_ref [test_name]
   ```
3. **Run Tests**:
   ```sh
   $ python -m pytest -v
   ```
   OR
   ```sh
   $ ./dev.py build pytest
   ```

### Creating Pull Requests

1. **Add a changelog entry**:
   - Add a new entry in the `CHANGELOG.md` file.
   - Follow the format: `## [Unreleased]` -> `## [Version]`.
2. **Commit Your Changes**:
   Your commit message should follow the [react convention](https://github.com/ubilabs/react-geosuggest/blob/main/CONVENTIONS.md).
   ```sh
   git add .
   git commit -m "feat(scope): add new feature"
   ```
3. **Push Your Branch**:
   ```sh
   git push origin feature/new-feature
   ```
4. **Create a Pull Request**: Go to the GitHub repository and create a pull request from your feature branch to `dev`.

### Creating a release

1.  Merge all your feature branches on `dev`
2.  Update `CHANGELOG.md`.
3.  Make a clean build in release mode (`./dev.py clean build -b Release`).
4.  Make sure everything works as intended and run test suite (`./dev.py pytest -b Release`). (`ctest` aren't working and are not to be used for now)
5.  Make sure the git repository has no work in progress
6.  Run the release script with `./dev.py -b Release release {bump-type}` with `bump-type` being either `major`, `minor` or `patch`
    if you want to modify either X, Y, or Z of the version number X.Y.Z
7.  add the updated CMakeLists.txt to git
8.  do a `git push --follow-tags`
9.  Merge master with `dev`

## Advanced Topics

### Logging

Holovibes uses a custom logging system with five log levels: Trace, Debug, Infos, Warnings, and Errors. Use the following syntax for logging:

```cpp
LOG_ERROR(logger_name, formated_string [,args]);
```

### Miscellaneous

#### Adding Elements to Front with QtCreator

1. Load the file `mainwindow.ui` in QtCreator. (or any other `.ui` file)
2. Add the widget and change its name.
3. Add a slot in `mainwindow.ui`.
4. Add the prototype in `MainWindow.hh` in the public/private slot and implement the function in `MainWindow.cc`.

#### Updating the UI from the API

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

## Troubleshooting

- If the app crashes after launch, try removing the app settings from `C:\Users\[user]\AppData\Roaming\Holovibes\[version]\*`.

Refer to [DEVELOPERS.md](DEVELOPERS.md) for detailed troubleshooting steps.

## Documentation

Generate the documentation using Doxygen:

```sh
$ ./dev.py doc
```

This will compile the documentation and open it in your default browser.

---

By following this tutorial, you should be able to set up your development environment, understand the project structure, adhere to coding standards, implement new features, test your code, and contribute effectively to the Holovibes project. Happy coding!
