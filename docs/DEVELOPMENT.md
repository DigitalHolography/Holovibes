Holovibes uses CMake presets and a pinned vcpkg manifest on Windows x64. The
application still requires CUDA 13.0 or newer; the dependency smoke test does not.

## First build

1. Install Git and Visual Studio 2022. Import the repository's `.vsconfig` in
   Visual Studio Installer. It selects the C++ workload, MSVC 14.44, Windows SDK
   10.0.26100.0 and CMake/Ninja. Keep the specified toolset installed side by side
   with newer versions. CMake 3.31 or newer is required.
2. Install a CUDA 13.0+ Toolkit compatible with the compiler and an NVIDIA driver
   suitable for that toolkit/GPU. No camera SDK is needed for the development preset.
3. In a terminal at the repository root, run:

   ```powershell
   .\dev.cmd build
   .\dev.cmd run
   ```

`dev.cmd` launches `dev.ps1` with a process-local execution policy, which also
works when ordinary PowerShell script execution is disabled. It does not change
the machine's execution policy or global PATH. Each invocation initializes its
own compiler environment. Python and Conan are not manual prerequisites for
building the application; vcpkg may download Python internally for dependency builds.

The first build without cached packages compiles Qt/OpenCV and can take a long
time. Subsequent builds reuse both the installed libraries and binary cache.
Use a reasonably short checkout path (for example `C:\src\Holovibes`) because
some upstream build tools have Windows path-length limitations.

## Commands and presets

| Command | Purpose |
| --- | --- |
| `dev.cmd configure` | Restore dependencies and configure development build |
| `dev.cmd build` | Configure if needed, then build RelWithDebInfo |
| `dev.cmd build -Preset windows-debug` | Build Debug separately |
| `dev.cmd build -Preset windows-release` | Build optimized Release with available cameras |
| `dev.cmd run -Preset windows-debug` | Run the selected build |
| `dev.cmd test` | Build and run C++ tests; GPU tests require working CUDA hardware |
| `dev.cmd dependencies` | Build and run the dependency smoke test without CUDA |
| `dev.cmd package` | Build Release and create the NSIS installer; requires NSIS |
| `dev.cmd clean -Preset windows-debug` | Remove only that preset's application build directory |

Build output is `out/build/<preset>/`. Libraries are in `out/vcpkg_installed/`,
and vcpkg itself is in `out/tools/vcpkg/`. All are ignored by Git. The install tree
is shared by presets; configure/build these presets sequentially when their
manifest features differ (for example switching testing on/off).

Extra arguments are forwarded to the selected configure/build/test command:

```powershell
.\dev.cmd configure -- -DHOLOVIBES_BUILD_CAMERAS=ON
.\dev.cmd build -- --parallel 4
```

`build` preserves options set by `configure`. Run `configure` again to reapply
preset defaults or change options; CMake automatically regenerates when its input
files change. Builds default to eight parallel jobs to limit memory usage from
large precompiled headers. The bootstrap also selects UTF-8 for the terminal so localized MSVC
output is parsed correctly by Ninja.

To use CMake directly, initialize the current terminal first:

```powershell
. .\scripts\bootstrap.ps1
cmake --preset windows-dev
cmake --build --preset windows-dev
```

If PowerShell blocks dot-sourcing, start a process-local session with
`powershell -NoProfile -ExecutionPolicy Bypass` first, or use `dev.cmd`.
For the smoke test use `. .\scripts\bootstrap.ps1 -DependenciesOnly`, then
configure/build/test the `windows-dependencies` preset. IDE users should run
bootstrap once to obtain vcpkg, then open the repository folder and select the
same preset. Machine-specific overrides belong in ignored `CMakeUserPresets.json`.

## Dependency versions and updates

The `builtin-baseline` in `vcpkg.json` pins both the vcpkg Git checkout and package
registry to the 2025.06.13 snapshot. This draft uses Qt 6.8.3, OpenCV 4.11.0,
HDF5 1.14.6, Boost 1.88.0 and nlohmann-json 3.12.0. GLM 0.9.9.8 and
spdlog 1.12.0/fmt 10.1.1 are explicitly retained to avoid changing application
math or custom logging code as part of the migration.

The manifest requests Qt Widgets/Charts/OpenGL/Network and image support, HDF5's
C++ binding, and OpenCV's Windows capture backends plus FFmpeg video encoding.
Unneeded Qt QML modules and OpenCV CUDA/DNN modules are not built. The smoke test
checks Qt Charts construction, an HDF5 data round trip, and MJPEG/MP4V encode/decode.

Update the baseline deliberately, run `dev.cmd dependencies`, and validate the
application and installer before accepting the update. Bootstrap refuses to
overwrite edits in its vcpkg checkout. Toolchain settings live in
`cmake/toolchain.json`; when changing MSVC, also update the preset's toolset and
`.vsconfig`. vcpkg keeps compiler/ABI checking enabled, so an incompatible cache
entry causes a rebuild rather than being reused.

## Binary cache

vcpkg automatically uses its local cache (normally
`%LOCALAPPDATA%\vcpkg\archives`). A second source can be supplied before any
`dev.cmd` command:

```powershell
$env:HOLOVIBES_BINARY_CACHE = 'files,\\build-server\holovibes-vcpkg,read'
.\dev.cmd build
```

An anonymous HTTP cache can use
`http,https://your-cache.example/{sha}.zip,read` instead. These are examples;
no remote cache endpoint or credentials are committed. Existing
`VCPKG_BINARY_SOURCES` settings are preserved. A trusted build machine can publish
to a shared folder with `VCPKG_BINARY_SOURCES=files,<absolute-path>,readwrite`.

The Windows CI workflow preserves its local binary cache with `actions/cache`
and uploads `vcpkg-windows-binary-cache` after trusted builds. Developers can
download that artifact from a successful workflow, extract it to a local folder,
and point `HOLOVIBES_BINARY_CACHE` at that folder using the `files` provider.
For continuous remote caching, configure the repository variable
`HOLOVIBES_BINARY_CACHE` with a read source and arrange publishing from trusted CI.
GitHub Packages/NuGet is also supported by vcpkg but requires authentication,
including for public packages. Never commit cache credentials.

## Cameras and deployment

`windows-dev`, `windows-debug`, and `windows-tests` disable camera plugins by
default; recorded files still work. Enable cameras with
`HOLOVIBES_BUILD_CAMERAS=ON`. Each plugin has an `AUTO`, `ON`, or `OFF` option:
`HOLOVIBES_CAMERA_IDS`, `HAMAMATSU`, `XIQ`, `XIB`, `OPENCV`, `BITFLOW`, `ADIMEC`,
`PHANTOM`, `PCO`, `ASI`, and `ALVIUM` (all share the `HOLOVIBES_CAMERA_` prefix).
`AUTO` builds available SDKs, `ON` requires the SDK, and `OFF` skips it.

SDK roots are CMake cache paths: `BITFLOW_SDK`, `EURESYS_SDK`, `PCO_SDK`,
`VIMBAX_SDK`, and `ASI_SDK`. Bundled vendor files remain under `Camera/libs`.
Changing the enabled camera set is best done in a separate preset/build directory.
The camera implementations and hologram processing algorithms are unchanged.
The only application source fix is a null check in the file reader during CLI
shutdown, after computation is marked stopped and before the reader is joined.

Qt is deployed once into `qt-runtime` and copied beside the development binary.
Runtime dependency scanning also collects CUDA and selected camera dependencies.
Windows system DLLs and NVIDIA driver DLLs are supplied by the operating system
and installed driver. IDS's optional delay-loaded `GpuAcc_64.dll` and `VCOMP90.dll`
are supplied by its camera driver installation. Installer staging uses the same runtime dependency rules:

```powershell
cmake --install out/build/windows-release --prefix out/stage
cpack --preset windows-release
```

The packaging preset uses zlib for faster installer creation. Use
`dev.cmd package -- -D CPACK_NSIS_COMPRESSOR=lzma` when the smaller installer is
worth the longer compression time.

## Python regression tests and legacy commands

Use a Python 3.11 virtual environment with `requirements.txt` for the existing
hologram regression suite. Its data uses Git LFS; fetch the data before running:

```powershell
git lfs pull
$env:HOLOVIBES_BIN = "$PWD\out\build\windows-dev\Holovibes.exe"
python -m pytest tests/test_holo_files.py
```

`python dev.py install/cmake/build/run/test/ctest/clean/preRelease` delegates to
the new presets. `-b dev`, `-b Debug`, and `-b Release` select the build mode.
`pytest`, `build_ref`, and version-bumping `release` remain available.
The previously ineffective `-i` path argument is rejected in favor of presets.
`clean` removes build output only; it no longer deletes regression output files.

## Draft validation

Validated locally with Visual Studio 2022/MSVC 14.44, Windows SDK 10.0.26100.0,
CMake 3.31.6, CUDA 13.0.2 and an RTX 4070 Ti SUPER:

- Release (including the five bundled camera plugins) and Debug build successfully.
- The dependency smoke test passes, including Qt Charts, HDF5 and AVI/MP4 codecs.
- All 93 dependency packages restore into an empty install directory using
  `--only-binarycaching`; cache restoration took 13 seconds, plus 5 seconds to install.
  The initial uncached dependency build took about 23 minutes on this PC.
- The 88 enabled C++ unit tests pass; two existing tests are disabled.
- CLI smoke checks repeat raw recording ten times, compare raw pixels with the
  input, and exercise hologram, AVI, MP4 and HDF5 moments exports with only Windows
  directories on PATH. This check is included in `dev.cmd test` and can also run
  against a staged application:

  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test-cli.ps1 -Executable out/stage/Holovibes.exe
  ```

- The combined CTest run passes 89 checks (88 unit tests plus the CLI check).
- `dev.cmd package` generates the NSIS installer, and its embedded CRC matches
  the generated file. The staged application passes CLI checks. Installing it on
  a clean Windows machine has not been tested.
- GUI startup passes using Qt's offscreen platform for ten seconds. Interactive
  display rendering and real camera acquisition still need hardware validation.
- The existing Python regression suite reports 57 skips because reference outputs
  are absent in this checkout. It has not verified numerical equivalence with a
  production build; obtain the established reference data before that comparison.

The CUDA toolkit used for these checks was extracted under ignored `out/tools/`
for validation only. It did not change the installed toolkit or driver. Developers
should install CUDA 13.0+ normally, or set `CUDA_PATH` to a compatible toolkit.

The hosted Windows workflow checks dependencies without a GPU. The optional
application job runs only on manual dispatch and needs a project-managed runner
with labels `self-hosted`, `Windows`, `X64`, `holovibes-cuda`, the declared toolchain,
an NVIDIA GPU/driver, and any desired camera SDKs. Provisioning that runner and a
shared cache service are separate infrastructure steps. The workflow has not yet
been run on GitHub; local validation covers the commands it invokes.

Before replacing a production build, validate Release and Debug, C++ and hologram
regression tests, GUI startup, video/HDF5 exports, the installed application on a
clean machine, and each camera used in the lab. Dependency smoke tests do not
replace those application/hardware checks.
