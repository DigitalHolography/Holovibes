## Changelog

### 14.8.2

- New display and contrast reticle layout.

- Revamped processholofiles.ps1.

### 14.8.1

- Revert changes on ametek camera BufferPartCount in ini files.

### 14.8.0

- New cuda texture class.
- The standard propagation idstance is now 480mm.
- New constrast refresh button.
- New documentation for ametek camera ini files.

### 14.7.2

- Fix ps1 standard preset.

### 14.7.1

- Add asi camera support.
- removed the stop fan button.

### 14.7.0

- fix convolution bugs.
- circular reticle overlay for contrast calculation.
- new presset.
- ps1 improvement.

### 14.5.6

- Button to toggle Ametek camera fans.

### 14.5.3

- Changement des valeurs par defaults pour le contraste

### 14.5.0

- Fix reticle bug in cli
- ui: Rec. -> Record
- fix avi file resolution
- split front and back in the repository

### 14.4.3

- Anamorphic image are interpolate for beeing square in .avi
- Removed duplicate \_L and \_R
- Readded benchmark mode in cli

### 14.4.2

- Fix:
  - Angular spectrum now working with a new lens. Allowing angular spectrum to work with anamorphic data.

### 14.4.1

- Fixed FFT frequencies computation
- Index 0 for z frequency now allowed
- Fixed Filter 2D crashing when no filter file provided
- Fixed recording and progress bar now shows frames being saved and being captured
- Fixed optimization issue with internal settings
- Updated S711 ini file
- Change batch_size from 64 to 32 in preset.json
- Corrected footer data in .holo files

### 14.4.0

- Nb. frames now isn't reset when interacting with certain UI components.
- Removed boundaries to Nb. frames, but it is set to the maximum when a file is loaded
- Rename UI elements:
  - "Registration" to "Image Registration"
  - "View" (next to Filter2D) to "Filter2D View"
  - "Regular Access" to "Sequential File Read" for the drop down menu of file mode loading
  - "Nb. img" to "Time Window"
  - "Renormalize" to "Renormalize Image Levels"
  - "Contrast" to "Brightness and Contrast"
- Change eye recorded:
  - Rename "Eye recorded" to "Tag"
  - There is a single button that switch between left, right or none
- Fixed image registration: Now working with anamorphic data (ex: 512x256).

### 14.3.0

- Updated all preset:
  - batchsize from 32 to 64
  - time_transformation_size from 16 to 32
  - time_transformation_stride 32 to 64
  - z width from 8 to 16
- Changed the UI option 'Load file in GPU' to a drop-down menu and renamed it to 'Load file in GPU VRAM'
- Added an option 'Load file in CPU RAM' in that drop-down
- Added UI buttons that specify which eye is recorded, which is reflected in the file name

### 14.2.0

- The UI can record again
- When closing a view, calculations of this view are stopped
- Record in ps1 do not overwrite a file with the same name
- When getting frames from the camera or the file memory transfer are now batched
- Input fps are calculated correctly
- Some UI related crash are now fixed (start = end index, 3d cuts, ...)

### 14.1.0

- Restucuration of the API

### 14.0.2

- Registration is fixed:
  - Now the registaration is working with LDH.
  - The registration does not need the fftshift to work anymore. Hence the image is registered whether or not the fftshift is activated.
- Add 3 new preset :
  - doppler_8b_512_256_25.json
  - doppler_8b_1024_768_120.json
  - doppler_8b_1280_800_130.json
- Rename doppler_8b_384_27.json to doppler_8b_384_384_27.json
- Rename doppler_8b_512_40.json to doppler_8b_512_512_40.json

### 14.0.1

- Remove the second .ps1 script
- Autocontrast works more consistently; thus removed the 'apply' button from the UI

### 14.0.0

- Create a second ps1 for registration
- Enhance the registration process:
  - The user can choose the radius of the circle where the cross-correlation is computed. A spinbox is added in the GUI.
  - The registration is now working in CLI mode.
  - The registration is now in the compute settings. The registration zone (circle radius) is saved in the compute settings.

### 13.10.0

- Add image registration. Used to stabilize the image to the center of the eye. A reference is taken when the user click on
  `registration` check box in the frontend.

### 13.9.0

- New dark theme style
- Add a boundary information above the focus
- Information worker data are now displayed in a table
- Add temperature for phantom cameras and alvium
- Add the possibility to use compute settings which doesn't have all the fields

### 13.8.0

- Fixed default S710 ini file
- Removed the BatchEnabled and the BatchInputFile in the UI
- Moved "CPU Record" from the main UI to the AdvancedUI.
- Mini panel now has contrast and camera settings
- Renamed "Record on GPU" to "Record queue on GPU"
- Renamed "FresnelTR" to "Fresnel Tr." and "AngularSP" to "Angular SP."
- All parameters for Filter2D and 3DCuts are now hidden when the checkbox are not enabled
- Show only active views in the window selection view dropdown
- Fixed reticle displaying white screen
- Removed useless Euresys eGrabber camera model option

### 13.7.5

- Renamed space transforms more accordingly: fft1 to Fresnel transform and fft2 to Angular spectrum
- New specifications table in S991, S710, S711 ini files
- BufferPartCount default parameter set to 128 instead of 64 for camera S711
- Added tooltips visible when putting mouse cursor on labels
- Fix install of camera Alvium
- Change the focus in mm instead of m in the main window

### 13.7.1

- Added the possibility to change .ini files before loading cameras
- Simplify .ps1 script
- Holovibes and the Process Holo Files script can now be installed separately in the installer
- Added a logo for the ps1 script
- Added moments

### 13.7.0

- Support for Alvium Camera
- Changed default preset to 'doppler_8b_384_27'
- Rename preset 'doppler_8b_384_28' to 'doppler_8b_384_27'
- Default display rate is now 24
- Added an executable desktop shortcut for ProcessHoloFiles.ps1
- Rename ini config variable 'NbImageBuffer' to 'BufferCountPart' for camera phantom_s710 / phantom_s711 / phantom_s991

### 13.5.1

- Adjusted S711 ini configuration file

### 13.5.0

- Create option in CLI for frame skip and fps for mp4 records
- Change the .ps1 to ask for frame skip (not for mp4 fps)
- Begin support for Alvium Camera
- Drop down menu to select any preset in the preset directory
- The ProcessHoloFiles.ps1 is now installed along Holovibes, in a separate directory
- Display the right input fps when it is indicated in the footer of the holo file

### 13.4.0

- Updates the documentation and merge the tutorial

### 13.3.2

- fixed maximum capacty of queues
- new camera dll

### 13.3.1

- fixed and added more presets
- fixed an issue when writing file name in light ui

### 13.3.0

- file path is now written with ' \ ' in the app
- ConfigUI: added ZSlider
- --fps option fixed
- cli now works properly for new holovibe's version footer
- mp4 file output is not hard set to 24 fps (avi isn't)
- record_mode isn't a compute_setting anymore
- default inputfps value is now 10000 (from 60)
- Added powershell script that allow user to :
  - select a folder with holo files
  - select an optionnal compute setting file (by default, takes it from each of the file's footer)
  - select an optionnal hololive exec (by default, takes it from the current folder)
  - select a file extention for the output
    It then output in the first choosen folder the cli output of each of the files inside.

### 13.2.3

- Fix a bug causing the overwrite of the output file
- Fix s710 and s711 camera .ini files
- Presets now have a number of frame to record
- Preset button added to ConfigUI
- Modifying buffer sizes when loading a file in GPU

### 13.2.2

- Presets fixed (again), removed auto contrast as it causes performance issue at the moment
- Tutorial added for installation and contribution in TUTORIAL.md
- LightUI: fixed "Idle" record progress bar state
- Output file path on selection cancelation fixed

### 13.2.1

- Presets fixed
- Ini files for s710 and s711 updated

### 13.2.0

- nb_frames_to_record & Record mode are now saved on quit
- Presets implemented
- When saving a file, it now automatically add the current date as a prefix (please have the clock of your computer correctly set)
- When selecting an existing file for its name, it'll automatically trim the date and number prefix and suffix.

### 13.1.2

- Keybind shortcut added:
  LightUI:
- Alt+M: open ConfigUI
- Ctrl+Q: Quit
  ConfigUI:
- Alt+C: open camera config file
- Alt+A: open advanced settings
- Ctrl+Shift+Q: reset settings and Quit
  Both:
- Ctrl+R: Start/Stop recording
- Ctrl+S: Browse Export File
- lightUI: minor ui change

### 13.1.1

- fix: In Configuration panel, focus spinbox fixed (navigating between values now works properly)

### 13.1.0

- nb_frame_record is now a user setting
- Contrast now on by default
- Changed "configurationUI" button location on lightUI upper settings
- LightUI pipeline buttons now are only active when a pipeline is active (prevents crash when pressing start button with no pipeline active)
- record progress bar now changes color both on info_panel and lightUI according to acquisition status
- record progres bar now always visible
- window now correctly resizable with start/stop button not changing in size
- Changing panel label again
- Mini panel UI layout changed
- record progress bar now change its label acording to its color
- 3d cuts are now compute settings and can be showed at the launch of the app
- z distance can no longer crash kernels when at 0.
- Fixed the spinbox bug in light ui.
- renamed propagation distance to focus
- changed duplicate filename nomenclature (from '-' to '\_')
- Compute settings gpu buffers aren't imported when loading a file to avoid surcharging the gpu
- Flatfield correction off by default in cameras ini files

### 13.0.O

- Record progres bar now always visible
- Window now correctly resizable with start/stop button not changing in size
- Changing panel label again
- Mini panel UI layout changed
- Record progress bar now change its label acording to its color

### 12.14.0

- Contrast now on by default
- nb_frame_record is now a user setting (saved upon closing)
- Changed "configurationUI" button location on lightUI upper settings
- LightUI pipeline buttons now are only active when a pipeline is active (prevents crash when pressing start button - with no pipeline active)
- record progress bar now changes color both on info_panel and lightUI according to acquisition status

### 12.13.1

- Add FlatFieldCorrection parameter to s711

### 12.13.0

- Fix crash/incorrect exit when trying to exit from light UI with a camera/file open
- Fix crash case on startup
- OutputFilePathLineEdit of lightUI is now corretly set on opening
- Start button now resets when the recording stops automatically

### 12.12.1

- Fix s711 camera

### 12.12.0

- New light UI added
- View windows now accessible and moveable
- Add log to inform when the record is going to stop

### 12.11.2

- Fix more record crashes
- push backs updated and doc added
- Convolutions are now correctly Saved and imported from compute settings
- View windows stay at the same position when reloaded
- CMakeList version updated

### 12.11.1

- Fix s711 camera
- Fix record crash
- Fix input queue size overflow
- 0 continuity record trully fixed
- Stops the record when all contiguous frames have been recorded

### 12.11.0

- Now possible to modify batch size in raw mode
- Fix cpu record crash
- Camera doesn't reload after record
- 710 camera works with 2 banks
- 0 continuity record fixed
- Color added in the info pannel to notify the user of performance issues.

### 12.10.5

- Fix automatic load for s991
- Fix input filter for Adimec 2A750

### 12.10.4

- Fix input filter crash : now possible to start in processed with camera open and input filter
- Fix 3d cuts contrast : XY contrast not affected, XZ and YZ now correctly loaded

### 12.10.3

- 8 bit recording and other camera settings fixed

### 12.10.1

- Improved memory management for 3d cuts
- 8 bit recording are still being fixed

### 12.10.0

- CPU record mode only activates when record is triggered
- Fix hflip zoom
- Fix record with no number of frames specified
- Handle Adimec Camera
- Fix camera double load
- Solve s710 stripe offset
- Solve user settings crash
- Fallback to default settings in case of faulty footer in file

### 12.9.9

- Only allocate the queues when the size is changed (no wait time when changing the image mode)

### 12.9.8

- Add the possibility to move the input queue on the cpu to record in cpu mode
- Add tests

### 12.9.7

- Fix testsuite
- Fix dev.py
- Add pseudo-benchmarks on testsuite

### 12.9.6

- Add s711 camera support
- Partial record fix : possible to record holograms and raw images (when indicating number of frames)

### 12.9.5

### 12.9.1

### 12.9.0

- Sticky windows

### 12.8.0

- Update camera and compute mode when closing

### 12.7.0

- Conan 2 toolchain

### 12.6.0

- Add s991 camera support
- Remove hue blue HSV option

### 12.5.0

- Remove HSV auto weight normalization

### 12.4.0

- Add different HSV treshold normalization methods

### 12.3.0

- Stop record when changing the record mode or the record queue size
- Ensure contiguity of the first _record queue size_ images recorded

### 12.2.0

- Add button too chose dynamically between CPU/GPU for the record queue

### 12.1.1

- Use variants instead of dynamic polymorphism for choosing the location of the queue (CPU/GPU)

### 12.1.0

- Preallocate the record queue

### 12.0.0

- Fix testuite
- Ensure temporal integrity of the record
- Put record_queue on CPU

### 11.9.0

- Add input_filter
- Add HSV coloring
- Fix RGB coloring

### 11.8.0

- Discall all changes beteween 11.5.0 and 11.7.0 included

### 11.4.0

- Add GPIO support with euresys frame grabbers
- Rework .ini files for s710

### 11.3.0

- Add high_framerate with euresys and phantom
- Use a config file inside of holovibes directly

### 11.2.0

- Fix logger flush on quit
- Import cine file
- Add setup_710

### 11.1.4

- Fix Endianness

### 11.1.3

- Fix load of holo with no footer

### 11.1.2

- Add compatibility with footer version 5
- Edit some names in the gui

### 11.1.1

- Fix Xiq camera
- Fix Bitflow_cyton camera
- Improve logger

### 11.1.0

- Update and clean loading of holo file using nlhomann
- Fix loading of holo file and gui
- Advanced window close on save
- Reset ComputeSettings in gui
- Refacto logging macro -> delete warning
- Add wrapper to std::chrono

### 11.0.1

- Add OpenCV compatibility
- Fix rotation function
- Fix rawbitshift

### 11.0.0

- Changed logger to spdlog (1.10.0 header_only)
- Add rule doxygen (1.9.4) documentation
- Upgrade Cmake from 3.22 to 3.24
- Upgrade zlib from 1.2.11 to 1.2.12
- Upgrade openssl from 1.1.1m to 1.1.1q
- Upgrade conan from 1.42.2 to 1.52.0
- Upgrade Jinja2 from 2.11.3 to 3.0
- Upgrade PyJWT from 1.7.1 to 2.4.0

### 10.7.2

- Fixed icons not being displayed properly
- Fixed cine not being able to be imported properly

### 10.7.1

- Removed compute descriptor from the code
- Fixed the color theme not working properly at the start
- User settings are now in .json format instead of .ini
- Reintroduced the record frame step in the advanced settings window
- Fixed crashes when enabling the reticle
- Fixed another crash when the reticle is enabled at the start of a .holo import
- Fixed crash on loading .holo with no footer
- CLI is now deterministic
- Fixed use of useless pipe in raw mode in CLI
- Input FPS in footer computed with 16 values
- CLI is contiguous and FPS default value is not constrained
- Fixed 3d cuts crash when desactivated and reactivated
- Fixed 3d cuts : automatically refresh the pipe on activation

### 10.7.0

- Global state refacto, needed to organize and gather up the global state (formerly in the compute descriptor) :
  - Creation of the global state holder (GSH), which puts rules on the flow of variables from the front to the backend (getters and setters), and centralizes the value specific state logic (like thresholds or constraints between variables)
  - Removal of the hundreds of std::atomic<> used everywhere there was some state flowing between workers, by :
    - mofifying the GSH only from the API, and the pipe's make request (to be modified)
    - providing caches to the ComputeWorker, in order to have a quick access to needed variables, and to be able to choose when to synchronize the pipe (a first step towards the removal of the request mechanism)
    - creating the FastUpdateHolder (has been implemented for a few releases already) for the high updated rate variables used by the InformationWorker

### 10.6.4

- Fixed a bug that prevented recording without crashing using the GUI
- The info panel dynamically changes size to fit the text if its too big for the normal layout

### 10.6.3

- Feat Add contiguous in footer.

### 10.6.2

- Feat ASW with Raw Bit shift
- Fix Batch stride changes does not affect record frame number
- Fix CLI Import file info
- Fix Record in Raw mode

### 10.6.1

- Advanced Settings window does not force the program to close after saving the changes
- Fix Batch input size not being set to 1 when using raw mode
- Added Input fps in footer HOLO v4

### 10.6.0

- Now compiles with Clang as well as MSVC
- Fixed a crash when stopping the import with 3Dcuts enabled
- Fixed a memory allocation problem with very large queues
- Fixed a visual bug with the record progress bar

### 10.5.2

- Fix --raw option
- Fix Camera ini opening
- Fix Advanced Settings Window size
- Fix Z_step initialization in ASW

### 10.5.1

- CLI remove --divide and --convolution (already present in compute_settings)
- Buffer default values changed

### 10.5

- New HOLO footer v4
- Add folder creation in AppData at runtime
- Compute settings from .ini format to JSON
- Upgraded CUDA from 11.2 to 11.5
- Solved some crashed linked to auxiliary windows
- Fixed a problem with the batch size not being initialized properly
- Moved the z boundary value to the Info panel
- Space transformation can now be changed while 3D cuts are enabled
- Some GUI elements are hidden until the corresponding image type or such is selected
- Changed a CLI argument from --ini to --compute-settings

### 10.4

- Upgraded from Qt 5.9.9 to Qt 6.2.1
- Installer now creates folder cameras_config in Appdata and put all cameras.ini in it
- Add record mode Cuts XY and Cuts YZ

### 10.3.1

- Fix Camera view in raw mode (the camera raw view wasn't displayed)
- Fix Compute settings loaded from compute_settings.ini
- Device security : Batch size and time stride not reset to low value
- Fix Load compute settings when no file/camera is running

### 10.3

- Remove Accu CheckBoxes (equivalent to accu_level == 0)
- Split Holovibes.ini in compute_settings.ini + global_settings.ini in %AppData% folder
- Remove Reset functionality
- Add new window AdvancedSettings to be able to change all parameters from UI
- Browse location of compute_settings.ini import/export

### 10.2

- Installer: creates a folder in AppData/Roaming
- Prevent crash when 3D cuts is active and space transformation is changed

### 10.1

- Changed BitFlow version: 6.40 -> 6.50

### 10

- Bump version

### 9.5

- Camera: bitflow is now a standalone camera mode
- CLI: fix batch size bug

### 9.4.1

- CLI: add load in GPU option

### 9.4

- Camera: add phantom 710 + bitflow camera (should be able to handle any camera with a bitflow framegrabber)
- Fast pipe (experimental): pipe speed up when batch size == time stride == time size

### 9.3.10

- CLI: fix segfault
- Holofile: fix exception when a field is empty

### 9.3.9

- CLI: load input file before .ini (avoid overriding parameters)

### 9.3.8

- cli: record total frames / time stride by default
- record: an arbitrary number of frames can now be skipped at the beginning of a recording
- cuda: check for compute capabilities

### 9.3.7

- Set Lambda default value to 852nm
- Fix Input Throughput display
- Fix unit test not working in WSL
- Automatic calculation of the total frames in relation to time stride to record the entire processed file
- Launching unit tests with release script

### 9.3.6

- Fix 3d cuts black bars
- Add ssa stft time transform

### 9.3.5

- Fix: Filter2D with anamorphic images
- Filter2D with CLI
- FFT2/Filter2D optimisation

### 9.3.4

- Add: Filter2D circles mask
- Fix: cuts and filter2D color overlays

### 9.3.2

- Fix: filter2D optimisation

### 9.3.1

- Fix: filter2D crash due to batch size
- Fix: filter2D view crash with 8bit files
- Fix: cine file number of frames
- CLI: better contrast handling

### 9.3

- Fix: CLI contrast and recording
- Fix: Camera frame copy
- GUI: Change names on frontend

### 9.2

- Filter2D: rework filter2D
- CLI: add progress bar and other info
- Fix: minor gui bugfixes
- Fix: batch_input_queue async copy and input file gpu_file_frame_buffer synchro

### 9.1

- Time Transform: add None mode
- Fix: stutters in raw view
- Fix: weird contrast spinbox behavior
- GUI: rename some labels
- CLI: rework cli to handle different types of files and load .ini config

### 9.0.5

- Cine: 10bit packed cine file support
- Fix: input queue size is not reduced when changing batch size a lot
- Fix: crash when importing a footer-less holofile
- GUI: Change sinbox scroll default values

### 9.0.4

- Cine file now read 12bit packed images

### 9.0.3

- PCA: fix horizontal bars bug when #img > 32

### 9.0.2

- Merge batch input queue
- Pre allocate cusolver work buffer

### 9.0

- Upgrade to 9.0 for final release

### 8.9.5

- GUI: remove useless lines at bottom

### 8.9.4

- GUI: disable batch_size spinbox while recording
- GUI: move convolve to image rendering panel
- Fix: overflow when correcting number of frames exported
- Fix: disable raw view while recording
- Fix: contrast on cuts while using reticle
- Fix: error window if cuda graphic card not detected
- Fix: do not launch frame record if batch size greater than record queue size

### 8.9.3

- Upgrade CUDA from 11.1 to 11.2

### 8.9.2

- GUI: disable composite when stopping computations
- GUI: disable options when selecting a camera not plugged
- Set default batch size to 1 to prevent crashes

### 8.9.1

- Camera: Handle properly Phantom S710 (using 2 banks)
- CUDA: on error show a dialog box and exit
- CUDA: use generic and optimized map operation for most kernels
- Allow saving avi and mp4 as square outputs
- GUI: Remove renomalize value and zstep
- Fix bugs related to composite mode and GUI
- Update docs and refactor code

### 8.9

- Export holograms to .avi and .mp4
- GUI: simplify export panel
- Add support for camera Phantom S710
- More error handling and bug fixes

### 8.8.1

- Fix crash when loading camera Adimec
- Fix HSV and crash when starting in composite mode

### 8.8

- Refactoring and bug fixes of threads
- Improve performance of renormalize and limit cuda device sync
- Apply autocontrast when updating renormalize
- Fixes in GUI and reticle

### 8.7

- Handle anamorphic images
- Add python scripts to convert .holo to .raw/.avi/.mp4 and .raw to .holo
- Implement integration test suite

### 8.6

- Implement CLI
- Small optimizations on convolutions, allocation of buffers
- Fix contrast on reticle zone

### 8.5.3

- Contrast on reticle zone if enabled
- GUI: simplified Filter2D
- Fix chart recording
- Renaming filter -> transformation

### 8.5.2

- Refactoring of file system
- Fix compatibility issue for chart feature

### 8.5.1

- Add new modes in chart
- Reduce default size of input_queue
- Fix raw queue allocation (synchronization between UI and backend)
- Fix memory leaks when opening raw windows

### 8.5

- Allow using a different batch size from stft step (decorrelate both parameters)
- Rename parameters and queue/buffers names for consistency (Direct -> Raw, stft -> time filter, SVD -> PCA)
- Fix raw recording: copy only required frames to raw queue, prevent skipping frames
- Improve computations pipe by freeing up resources before new allocations
- Clean GUI (remove Postprocessing panel: move chart in Export panel and convolution in View panel)
- More error handling (queues, CUDA not detected)

### 8.4.1

- Remove dead code and unused functions
- Update docs: README, CONTRIBUTING and DEVELOPERS

### 8.4

- Fix enqueue multiple (images in wrong order)
- Fix image accumulation composite
- Fix unwrap 2D (bad plan)
- Fix autocontrast with stft cut
- Add unit tests for Queue
- Clean project (remove trash files, move directories, remove unused cameras...)
- Remove DLL warnings

### 8.3.1

- Apply autocontrast automatically
- Renormalize enabled by default and moved in view panel in GUI
- Add more triggers mode for camera Hamamatsu
- GUI: deactivate lens and raw view buttons when the windows are closed
- GUI: open documentation page directly in web browser
- Fix bug when using camera Xiq
- Fix bug when switching focus between windows too quickly
- Fix bug with average plot and stft cuts
- Fix bug with average plot if selected zone is empty
- Fix bug when modifying space filter with lens view opened
- Fix bug when modifying compute settings while doing a convolution
- Fix bug when using image acc in composite mode
- Prevent setup installer to restart silently

### 8.3

- Improve performances/reliability of thread reader and allow loading the entire input file in GPU for really high FPS
- Show camera input FPS in Info panel
- Auto-scale for average signal plot
- Clean GUI of post-processing (convolution and average signal) and reduce size of main window
- Remove ROI save/load, record frame step from GUI and square_pixel, scale bar
- Put main window size and max stft cuts windows size in holovibes.ini
- Fix STFT cuts work with SVD mode
- Fix phase unwrap 2d
- Fix crash in direct mode with raw view enabled
- Setup installer do not run Holovibes at the end and installs C++ redistribuable 2019 if not already installed

### 8.2.1

- Fix raw view and raw recording

### 8.2

- Batch beginning of pipeline
- Remove frame normalization
- Implement unit test suite
- Change translation step from 1/10 to 1/20
- Remove useless parameters from import panel (width, height, ...)
- Refactor of autocontrast and image accumulation code

### 8.1

- Fix small bugs preventing composite HSV and RGB images to render correctly
- Remove features related to motion and focus (crop STFT, zoom lock, stabilization, interpolation, jitterm zernike, aberration, autofocus)
- Rename class attributes for classes ComputeDescriptor and FrameDescriptor by cd*and fd* for consistency
- Clean code of the pipe and add some comments
- Clean and split README in multiple files: AUTHORS, CHANGELOG, CONTRIBUTORS, DEVELOPERS and TODO

### 8.0

- Enable p_acc for svd
- Reduce timeout when csv recording (seems to avoid crashes)

### 7.9.1

- Add constant for renormalization
- Some more Filter2D fixes

### 7.9

- Fix SVD
- Fix Filter2D
- Remove filename info when batch recording

### 7.8.3

- Remove some convolution kernels
- Pre allocate SVD buffers
- Remove \_cmake from installer path

### 7.8.2

- Fix filter2D

### 7.8.1

- Fix SVD, it actually displays images now
- Remove useless filename informations when recording

### 7.8

- (WIP) Add SVD as a time filter mode
- stft_steps is now in holovibes.ini
- Camera now use FRAME_TIMEOUT from holovibes.ini
- Change CMakeLists.txt to make it compatible with CMake 3.16.1

### v7.7

- Camera PixelLink
- Some camera debug

### v7.6.2

- Stabilization: improve stabiliztaion algorithm (cross correlation)
- Add image normalization option in postprocessing
- Add different filter2D options (low/high/band pass)

### v7.6.1

- Holovibes can now handle rectangular input
- Quick convolution kernel selection
- Python scripts to convert .raw and old .holo files to new .holo files
- Fix .holo files not loading if json footer is empty

### v7.6

- New 64 bytes holo file header with endianness
- Fix overlay selection bug on rectangular windows
- Fix crash on filter2D

### v7.5.1

- Camera Hamamatsu bugfixes and updates
- Window ratio bugfix
- CMake does not try to re-build the project every time

### v.7.5

- Reduce output queue size when in raw output mode
- Number of recorded frames has a step (default=1024)
- When recording if a file with the same name already exists it won't be overwritten
- Set the amount of bitshift in direct mode to compensate for cameras recording on a weird number of bits
- Camera hamamatsu: cleanup and bugfixes

### v.7.4

- Reticle now handled as an overlay instead of a shader
- Reticle scale can now be controlled from a spinbox on the main window
- Contrast inversion
- z_step now defaults to 0.005
- Saving throughput in MB/s in the info manager when recording
- Fix not being able to open large holo files due to an overflow
- Fix a crash / freeze when opening a file with parameters overloading the GPU
- Fix holographic mode not triggering when changing algorithm from None to FFT

### v.7.3

- Toggle reticle on view window
- Scale reticle with "reticle_scale" in Holovibes.ini
- Fix holo file segfault

### v.7.2

- CMake build
- .holo file format
- Bugfixes on Xib camera
- General code cleanup

### v.7.1

- add: dark mode for chart window
- up: replace std::cout calls by logger class calls
- add: logger class
- up: Change chart autoscale offset
- up: Remove qwt library, plot now uses QtCharts
- fix: Use std::filesystem instead of std::experimental::filesystem
- up: Remove complex mode
- up: remove \_.ccincludes for moc\_\_ files
- up: remove vibrmetry
- up: remove flowraphy
- up: fix build ad Qt moc file generation
- up: use QGuiAppication for screen geometry
- up: remove stftlongtimes
- fix: .cu fix arument type warning with static_casts
- up: composite.c / hsv.cu remove debug gpu function causing warnings
- fix: array.hh postprocessing.cc fix initialize type warnings
- fix: tools.cu /MainWindow.cc fix initialize type warnings
- up: compute_desriptor.hh fix std::atomic warnings + code format
- up: bump CUDA vrsion to 10

### v.7.0

?

### v.6.7

- fix: plot is now on signal on launch
- fix: hsv h part on convolution

### v.6.6

- add doc stft longtimes
- add: icon

### v.6.5

- up:stft longtimes is now only on magnitude
- fix: imgtype can't change when stft longtimes is checked

### v.6.4

- fix: rainbow hsv is now in stftcuts
- up: pmin and pmax are now copied from rgb to and from hsv
- fix: can't do cropped stft and stft longtimes at the same time
- add: convolution on composite image

### v.6.3

- fix: convolution bug due to auto contrast
- fix: for stft longtimes p and pacc are dependent of #img 2
- fix: camera window size

### v.6.2

- cuda v10
- fix: main window doesn't show stft longtimes
- fix: p maximum is dependent of pacc
- up: convolution kernel is now circshift

### v.6.1.2

- up: fixed frames of xz yz cuts when longtimes
- fix: pacc limit and remove composite in stft longtimes

### v.6.1.1

- add: new mode stft longtimes
- fix: pixel size not updated
- add: first holovibe testsuite

### v.6.1

- up: hsv should be fine now (blur added)

### v.6.0.9

- fix: corrected hue part of HSV
- add: can divide by convolution
- add: renormalize image after convolution
- fix: convolution works
- fix: optimized reading of convolution kernel

### v.6.0.8

- add: HSV representation

### v.6.0.7

- fix: Add of the automatic set of zmin et zmax for the autofocus
- fix: initializing of a file holovibes.ini if it doesn't exist
- add: bfml files for adimec camera
- add: direct and algorithm none are now rectangular
- fix: Change the step for the plot auto scale to avoid strange behaviors

### v.6.0.6

- up: update MainWindow.cc to add documentation
- add: documentation first steps with holovibes
- add: normalization in image rendering panel

## September 2018 - December 2018

Ellena D., Clement F., Danae M. and Hugo V. are now working on the project until December 2018.

### v.6.0.5

- fix: average computation in plot

### v.6.0.3

- xiq: potential fix for error on closing (not tested with the camera)
- direct_refresh: commented the function
- fix: keep scale and translation when changing image type
- updated doxygen

### v.6.0.2

- fix: restored direct refresh function, still needed for direct recording

### v.6.0.1

- add: record synchronized with input file.
- add: negative m for zernike

### v.5.9.0

- change: show all tabs by default except MotionFocus and PostProcessing
- fix: nSize limited to 128 when reading file
- fix: zernike polynomials now correctly called and take into account the factor

### v.5.8.0

- add: possibility to change phase Number during stft cuts
- camera: fixed XIB closing errors
- aberrations: front end
- aberration: fft done using a temporary buffer
- add: cuda error checks after fft c2c
- remove: useless fields of ini file
- gui: changed titles and layouts

### v.5.7.4

- add: zoom lock in reset transform
- add: drag&drop import path

### v.5.7.3

- fix: Now using windows 10 sdk
- add: cuda 9.1

### v.5.7.2

- add: displaying of the output queue in the info tab

### v.5.7.1

- fix: disable whole AutofocusGroupBox instead of just the 'Run Autofocus' PushButton
- add: main doxygen page

### v.5.7.0

- add: rainbow_overlay
- fix: kernel_composite /(red - blue)
- fix: wheel event is only handled in direct window now.
- add: RGB weight add: precompute colors to accelerate XY composite

### v.5.6.7

- add: now resizing to original size when we uncheck square_pixel
- remove: frameless SliceWindow
- add: raw view in hologram.
- add: recording raw while we are in hologram mode
- fix: raw recording available in slices

### v.5.6.6

- change: thread_sync: added a generic way to run a function on mainwindow in the right thread
- fix: slices are now closed properly. Crop stft is now working with stft cuts.
- fix: prevent slice window from moving when resizing
- fix: camera hamamatsu uses the same sdk than the others (8.1)
- fix: resized correctly when passing from composite to direct mode
- add: minimum size to SliceWindow
- add: frameless slice window
- add: resizing slice window automatically remove square_pixel

### v.5.6.4

- fix: autocontrast_slice request added to have autocontrast on both windows at the same time.

### v.5.6.3

- fix: right click doesn't dezoom in slice anymore
- fix: 24-bit recorder is now working properly.

### v.5.6.2

- add: composite_area: front end to select the area
- fix: recording_24: writing the correct bytes
- fix: divided InputThroughput by stft_steps change: 4 decimals in Scale Bar correction SpinBox
- fix: cudav9 used in camera utils

### v.5.5.5

- change: stft steps maximum value set to 2^31
- add: progress bar: bar showing the progress in the input file
- fix: interpolation: added the categori in ini file

### v.5.5.4

- fix: camera: xib: better default values
- add: Hamamatsu camera
- add: stft is now limited to the zoomed area
- fix: now direct mode don't do any computation (issue direct_refresh)
- fix: autofocus and fft/filter2d have now their own classes managing their buffers. lot of code simplifying.
- add: class contrast used to handle the autocontrast part of the pipe
- fix: fft shift and log are now in the contrast class
- add: converts class containing convertion complex to float + that will contain float to short
- fix: free in correct destructors
- fix: composite images fix : stft*env* passed instead of stft_buf

### v.5.5.3

- fix: memory_allocation: use unique_pointers for some queues
- fix: raii: moving gpu buffers to unique pointers (WIP)
- fix: raii: removed most raw memory management (on CPU side)
- fix: raii: removed some raw pointers
- fix: since we don't disable stft while closing window anymore, we need to check in the pipe::exec if we're in direct mode (we do not want to take steps into account in direct)

### v.5.3.0

- Add: XY stabilization
- Fix: Overlay overhaul
- Add: displaying slice in filename recorded
- Remove: average plot in ini file
- Remove: Selection of the number of bits in records. Now selecting automatically
- Add: XY Zoom in stft view mode
- Add: Now Using Visual Studio 2017, Cuda 9.0 and Boost 1.65.1
- Fix: GUI overhaul
- Fix: Code cleaning

### v.5.2.2

- Fix: PhotonFocus camera bug on first frames

### v.5.2.2

- Add: PhotonFocus camera

### v.5.2.0

- Add: Composite image
- Add: Autofocus reworked, working in STFT
- Add: lens displayed as complex numbers

### v.5.1.1

- Add: FFT shift enable automatically after filter2D selection
- Add: developper names
- Add: Change output type checkboxes into a comboBox

- Remove: average enabled in holovibes.ini
- Remove: 32bits float output type

### v.5.1.0

- Add: black bar in fullscreen mode
- Add: averaging over x, y and p range and displayed by red overlay on each view
- Add: image accumulation for each view (XY, XZ and YZ) independently

- Remove: External trigger

- Fix: 2D filter selection direction was inversed when cursor was not in the bottom-right quadrant
- Fix: Both ',' and '.' are supported as decimal separator

## September 2017 - December 2017

Eloi C., Florian L., and Julien G. are now working on the project until December 2017.

### v.5.0.0

- Update: Qt 5.8 -> 5.9
- Update: STFT is enabled by default
- Add Beta feature: 3D view
- Phase accumulation
- Add: STFT cuts:
  - Record
  - Image type (Modulus / Squared Modulus / Argument / Complex)
- Full screen
- External Trigger (STFT)
- Update: Rotation and Flip mechanism

### v.4.3.0

- Add: STFT cuts :
  - Cross Cursor
  - Auto contrast at start
- Add: FocusIn event to select windows in program
- Add: Direct output buffer

### v.4.2.0

- Add: STFT cuts :
  - Rotation and flip
  - Contrast
  - Log Scale
  - Spacial images accumulations
- Add: title detect for imported files
- Add: New colors for Autofocus, 2DFilter, and Average when selected
- Update: Record now outputs a raw with properties in title readable by title detect feature
- Update: OpenGL now using Dynamic pipeline
- Update: Pop-ups are now displayed in info bar
- Several bugs fixed

## January 2017 -- July 2017

Alexandre B., Thomas J. are now working on the project until July 2017.

### v.4.0.0

- Bug fixing
- Jump a version to match CNRS repository (4.0)

### v.2.5.0

- Bug fixing
- Errors in Holovibes are now displayed & better handled.

### v.2.4.0

- STFT has been rewritten completely with great result improvement
- Thread_reader as been completely rewritten.
- 2D_unwrap has been rewritten.
- Every dll has been updated : Hovovibes is now in 8.0.
- Add Reference substract from the current image (sliding and settled)
- Add Filter2D feature.
- Add Complex display & record

### v.2.3.0

- FFT1 & FFT2 fixed
- FFT1D temporal performances greatly improved
- add accumulation stack feature
- fix several bugs

### v.2.2.0

- Add float file import
- Add complex file write/import
- Add complex color display
- several GUI changes

### v.2.1.0

- Add: demodulation mode
- Add: Special queue to handle Convolution & Flowgraphy
- Add: Convolution with loaded kernels
- Add: temporal Flowgraphy
- Add: cine file parsing
- Add: GUI themes

### v.2.0.1

- Several bugs fixed

## September 2016 -- January 2017

Cyril, Cl�ment are now working on the project until January 2017.

### v.1.5.0

- Camera Adimec can now run up to 300Hz (instead of the previous 60Hz)
- Holovibes can still be launched if visa.dll is not installed as it is now loaded only when needed
- Holovibes can unwrap in 6 different ways the images
- New part in the GUI, displaying information like the framerate
- Dependencies are now handled in a smarter way (not including things where they should not be)
- Ini file is now more dense and can configure more parameters

### v.1.4.0

- Pipeline has been renamed to Pipe, and Pipeline now refers to a another way to treat buffers, with several actions being done at the same time by workers

### v.1.3.0

- GPIB is now working with VISA, handling several instructions per blocks
- Minor changes

### v.1.2.0

- Adimec Quartz A2750 camera is now supported
- Stft can be applied to the image (instead of fft1/fft2)

### v.1.1.0

- The project works on CUDA 7.5
- The project works on QT 5.5
- Edge camera is now heavily handled
- XIQ now supports ROI

### v.1.0.2

- The autofocus can now be done recursively, allowing it to be performed must faster.
- Batch ouput can be done as float.
- Zoom is properly working now.
- The interface is responsive

### v.1.0.1

- Instead of taking the source images directly from the camera, the user can specify an input file (.raw) which will be read by the program.
- The autofocus has been improved a lot in terms of precision.
- Some small performance changes.

## September 2015 -- January 2016

Eric, Arnaud and Geoffrey are now working on the project until January 2016.

### v1.0.0

Major release

- Add documentation `;-)`
- Add HoloVibes' icon

### v0.7.4

- Minor GUI fix

### v0.7.3

- GUI average/ROI plot and record bug fix
- Minor GUI fix

### v0.7.1

- All records can now be stopped
- Autofocus is now handling selection zones
- Minor GUI changes

### v0.7

- Add autofocus algorithm (using global variance and average magnitude - sobel)
- Some minor fixes
- update to qwt 6.1.2
- update to qt 5.4

### v0.6.1

- CSV plot and batch simultaneously

### v.0.6.0

- CSV batch

### v.0.5.9

- Improved DLL loader
- Add smart pointers (memory safety)

### v.0.5.8

- GUI bind gpib
- GUI minor bugs fixed
- Recorder bug fixed
- Camera DLL loader

### v.0.5.7

- GUI ROI plot is freezing when recording
- GUI changing ROI plot number of points on the fly

### v.0.5.6

- GUI plot average can be resized
- GUI auto scale for plot average
- GUI minor bugs fixed

### v.0.5.5

- GUI plot average

### v.0.5.4

- GUI/CLI minor changes
- Update iXon camera

### v.0.5.3

- GUI bug fixes
- Add average values recording

### v.0.5.2

- Fix FFT shift corner algorithm

### v.0.5.1

- Fix dequeue method for Recorder

### v.0.5

- Zone selection (OpenGL)
- Zoom (OpenGL)
- Signal/Noise zone selection average
- Auto loading/saving Holovibes' configuration in INI file
- Minor bug fixes
- New PCO cameras handling (+ pco.edge, camera checking)
- Auto-contrast
- Camera Andor iXon support (beta)

### v.0.4.3

- GUI
  - Visibility methods changed to handle both case instead of having two methods
  - SpinBox sizes changed
  - Names of windows and parameters changed
  - When log scale is enabled, the value given to the pipeline is not powered to 10^9
  - When contrast enabled, enabling the log scale will update the contrast (to convenient values).
- Update CLI
  - Display mode is not available in --nogui mode
  - No Qt in CLI mode

### v.0.4.2

- Add Qt user interface.
  - Keyboard shortcuts
  - New Qt OpenGL window (resizeable)
  - Updates holograms parameters in live
    - Guards and protections against bad user actions
    - Camera change on the fly
    - Record canceling feature
- Add pipeline to apply algorithms.
- Fix queue issue when using big endian camera (Queue ensures that his content is little endian).
- Better memory management (less cudaMalloc), resources are allocated once at start and only few reallocations occurs when tweaking the Phase# parameter.
- Shift corners algorithm has been optimized.
- Make contiguous has been improved (no cudaMalloc and cudaMemcpy).
- Add manual contrast correction.
- Add log10 filter.
- Post-FFT algorithms are more precise (using floats instead of ushorts).
- Thread shared resources are no longer allocated in threads.
- CUDA kernels are less monolithic (complex-real conversions are separated).
- Fix pixelfly hologram mode.
- CLI updated (--nogui mode)
- CLI can set parameters on GUI.

### v.0.3.0

- Add FFT2 algorithm written in CUDA.
- Images to be displayed remains in GPU memory (no more copy to CPU-GPU).
- Fix queue endianness handling.
- 8-bit frames are rescaled in 16-bit frames before computation.

### v.0.2.0

- Add FFT1 algorithm written in CUDA.
- When FFT is enabled, the recorder writes output frames.
- When FFT is enabled, the display shows output frames at 30fps.
- Some bug fixes.

### v.0.1.1

- Fix recorder issue.
- Fix pike 16-bit issue.

### v.0.1

- Command line front-end.
- Handles 4 cameras: xiq, ids, pike, pixelfly.
- Use INI files to configure cameras.
- OpenGL display.
- Records frames.
