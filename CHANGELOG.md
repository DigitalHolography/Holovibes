## Changelog

### XXX

* CLI Fixed --raw opion

### 10.5.1

* CLI remove --divide and --convolution (already present in compute_settings)
* Buffer default values changed

### 10.5

* New HOLO footer v4
* Add folder creation in AppData at runtime
* Compute settings from .ini format to JSON
* Upgraded CUDA from 11.2 to 11.5
* Solved some crashed linked to auxiliary windows
* Fixed a problem with the batch size not being initialized properly
* Moved the z boundary value to the Info panel
* Space transformation can now be changed while 3D cuts are enabled
* Some GUI elements are hidden until the corresponding image type or such is selected
* Changed a CLI argument from --ini to --compute-settings

### 10.4

* Upgraded from Qt 5.9.9 to Qt 6.2.1
* Installer now creates folder cameras_config in Appdata and put all cameras.ini in it
* Add record mode Cuts XY and Cuts YZ

### 10.3.1

* Fix Camera view in raw mode (the camera raw view wasn't displayed)
* Fix Compute settings loaded from compute_settings.ini
* Device security : Batch size and time stride not reset to low value
* Fix Load compute settings when no file/camera is running

### 10.3

* Remove Accu CheckBoxes (equivalent to accu_level == 0)
* Split Holovibes.ini in compute_settings.ini + global_settings.ini in %AppData% folder
* Remove Reset functionality
* Add new window AdvancedSettings to be able to change all parameters from UI
* Browse location of compute_settings.ini import/export

### 10.2

* Installer: creates a folder in AppData/Roaming
* Prevent crash when 3D cuts is active and space transformation is changed

### 10.1

* Changed BitFlow version: 6.40 -> 6.50

### 10

* Bump version

### 9.5

* Camera: bitflow is now a standalone camera mode
* CLI: fix batch size bug

### 9.4.1

* CLI: add load in GPU option

### 9.4

* Camera: add phantom 710 + bitflow camera (should be able to handle any camera with a bitflow framegrabber)
* Fast pipe (experimental): pipe speed up when batch size == time stride == time size

### 9.3.10

* CLI: fix segfault
* Holofile: fix exception when a field is empty

### 9.3.9

* CLI: load input file before .ini (avoid overriding parameters)

### 9.3.8

* cli: record total frames / time stride by default
* record: an arbitrary number of frames can now be skipped at the beginning of a recording
* cuda: check for compute capabilities

### 9.3.7

* Set Lambda default value to 852nm
* Fix Input Throughput display
* Fix unit test not working in WSL
* Automatic calculation of the total frames in relation to time stride to record the entire processed file
* Launching unit tests with release script

### 9.3.6

* Fix 3d cuts black bars
* Add ssa stft time transform

### 9.3.5

* Fix: Filter2D with anamorphic images
* Filter2D with CLI
* FFT2/Filter2D optimisation

### 9.3.4

* Add: Filter2D circles mask
* Fix: cuts and filter2D color overlays

### 9.3.2

* Fix: filter2D optimisation

### 9.3.1

* Fix: filter2D crash due to batch size
* Fix: filter2D view crash with 8bit files
* Fix: cine file number of frames
* CLI: better contrast handling

### 9.3

* Fix: CLI contrast and recording
* Fix: Camera frame copy
* GUI: Change names on frontend

### 9.2

* Filter2D: rework filter2D
* CLI: add progress bar and other info
* Fix: minor gui bugfixes
* Fix: batch_input_queue async copy and input file gpu_frame_buffer synchro

### 9.1

* Time Transform: add None mode
* Fix: stutters in raw view
* Fix: weird contrast spinbox behavior
* GUI: rename some labels
* CLI: rework cli to handle different types of files and load .ini config

### 9.0.5

* Cine: 10bit packed cine file support
* Fix: input queue size is not reduced when changing batch size a lot
* Fix: crash when importing a footer-less holofile
* GUI: Change sinbox scroll default values

### 9.0.4

* Cine file now read 12bit packed images

### 9.0.3

* PCA: fix horizontal bars bug when #img > 32

### 9.0.2

* Merge batch input queue
* Pre allocate cusolver work buffer

### 9.0

* Upgrade to 9.0 for final release

### 8.9.5

* GUI: remove useless lines at bottom

### 8.9.4

* GUI: disable batch_size spinbox while recording
* GUI: move convolve to image rendering panel
* Fix: overflow when correcting number of frames exported
* Fix: disable raw view while recording
* Fix: contrast on cuts while using reticle
* Fix: error window if cuda graphic card not detected
* Fix: do not launch frame record if batch size greater than record queue size

### 8.9.3

* Upgrade CUDA from 11.1 to 11.2

### 8.9.2

* GUI: disable composite when stopping computations
* GUI: disable options when selecting a camera not plugged
* Set default batch size to 1 to prevent crashes

### 8.9.1

* Camera: Handle properly Phantom S710 (using 2 banks)
* CUDA: on error show a dialog box and exit
* CUDA: use generic and optimized map operation for most kernels
* Allow saving avi and mp4 as square outputs
* GUI: Remove renomalize value and zstep
* Fix bugs related to composite mode and GUI
* Update docs and refactor code

### 8.9

* Export holograms to .avi and .mp4
* GUI: simplify export panel
* Add support for camera Phantom S710
* More error handling and bug fixes
### 8.8.1

* Fix crash when loading camera Adimec
* Fix HSV and crash when starting in composite mode

### 8.8

* Refactoring and bug fixes of threads
* Improve performance of renormalize and limit cuda device sync
* Apply autocontrast when updating renormalize
* Fixes in GUI and reticle

### 8.7

* Handle anamorphic images
* Add python scripts to convert .holo to .raw/.avi/.mp4 and .raw to .holo
* Implement integration test suite

### 8.6

* Implement CLI
* Small optimizations on convolutions, allocation of buffers
* Fix contrast on reticle zone

### 8.5.3

* Contrast on reticle zone if enabled
* GUI: simplified Filter2D
* Fix chart recording
* Renaming filter -> transformation

### 8.5.2

* Refactoring of file system
* Fix compatibility issue for chart feature

### 8.5.1

* Add new modes in chart
* Reduce default size of input_queue
* Fix raw queue allocation (synchronization between UI and backend)
* Fix memory leaks when opening raw windows

### 8.5

* Allow using a different batch size from stft step (decorrelate both parameters)
* Rename parameters and queue/buffers names for consistency (Direct -> Raw, stft -> time filter, SVD -> PCA)
* Fix raw recording: copy only required frames to raw queue, prevent skipping frames
* Improve computations pipe by freeing up resources before new allocations
* Clean GUI (remove Postprocessing panel: move chart in Export panel and convolution in View panel)
* More error handling (queues, CUDA not detected)

### 8.4.1

* Remove dead code and unused functions
* Update docs: README, CONTRIBUTING and DEVELOPERS

### 8.4

* Fix enqueue multiple (images in wrong order)
* Fix image accumulation composite
* Fix unwrap 2D (bad plan)
* Fix autocontrast with stft cut
* Add unit tests for Queue
* Clean project (remove trash files, move directories, remove unused cameras...)
* Remove DLL warnings

### 8.3.1

* Apply autocontrast automatically
* Renormalize enabled by default and moved in view panel in GUI
* Add more triggers mode for camera Hamamatsu
* GUI: deactivate lens and raw view buttons when the windows are closed
* GUI: open documentation page directly in web browser
* Fix bug when using camera Xiq
* Fix bug when switching focus between windows too quickly
* Fix bug with average plot and stft cuts
* Fix bug with average plot if selected zone is empty
* Fix bug when modifying space filter with lens view opened
* Fix bug when modifying compute settings while doing a convolution
* Fix bug when using image acc in composite mode
* Prevent setup installer to restart silently

### 8.3

* Improve performances/reliability of thread reader and allow loading the entire input file in GPU for really high FPS
* Show camera input FPS in Info panel
* Auto-scale for average signal plot
* Clean GUI of post-processing (convolution and average signal) and reduce size of main window
* Remove ROI save/load, record frame step from GUI and square_pixel, scale bar
* Put main window size and max stft cuts windows size in holovibes.ini
* Fix STFT cuts work with SVD mode
* Fix phase unwrap 2d
* Fix crash in direct mode with raw view enabled
* Setup installer do not run Holovibes at the end and installs C++ redistribuable 2019 if not already installed

### 8.2.1

* Fix raw view and raw recording

### 8.2

* Batch beginning of pipeline
* Remove frame normalization
* Implement unit test suite
* Change translation step from 1/10 to 1/20
* Remove useless parameters from import panel (width, height, ...)
* Refactor of autocontrast and image accumulation code

### 8.1

* Fix small bugs preventing composite HSV and RGB images to render correctly
* Remove features related to motion and focus (crop STFT, zoom lock, stabilization, interpolation, jitterm zernike, aberration, autofocus)
* Rename class attributes for classes ComputeDescriptor and FrameDescriptor by cd_ and fd_ for consistency
* Clean code of the pipe and add some comments
* Clean and split README in multiple files: AUTHORS, CHANGELOG, CONTRIBUTORS, DEVELOPERS and TODO

### 8.0

* Enable p_acc for svd
* Reduce timeout when csv recording (seems to avoid crashes)

### 7.9.1

* Add constant for renormalization
* Some more Filter2D fixes

### 7.9

* Fix SVD
* Fix Filter2D
* Remove filename info when batch recording

### 7.8.3

* Remove some convolution kernels
* Pre allocate SVD buffers
* Remove _cmake from installer path

### 7.8.2

* Fix filter2D

### 7.8.1

* Fix SVD, it actually displays images now
* Remove useless filename informations when recording

### 7.8

* (WIP) Add SVD as a time filter mode
* stft_steps is now in holovibes.ini
* Camera now use FRAME_TIMEOUT from holovibes.ini
* Change CMakeLists.txt to make it compatible with CMake 3.16.1

### v7.7

* Camera PixelLink
* Some camera debug

### v7.6.2

* Stabilization: improve stabiliztaion algorithm (cross correlation)
* Add image normalization option in postprocessing
* Add different filter2D options (low/high/band pass)

### v7.6.1

* Holovibes can now handle rectangular input
* Quick convolution kernel selection
* Python scripts to convert .raw and old .holo files to new .holo files
* Fix .holo files not loading if json footer is empty

### v7.6

* New 64 bytes holo file header with endianness
* Fix overlay selection bug on rectangular windows
* Fix crash on filter2D

### v7.5.1

* Camera Hamamatsu bugfixes and updates
* Window ratio bugfix
* CMake does not try to re-build the project every time

### v.7.5

* Reduce output queue size when in raw output mode
* Number of recorded frames has a step (default=1024)
* When recording if a file with the same name already exists it won't be overwritten
* Set the amount of bitshift in direct mode to compensate for cameras recording on a weird number of bits
* Camera hamamatsu: cleanup and bugfixes

### v.7.4

* Reticle now handled as an overlay instead of a shader
* Reticle scale can now be controlled from a spinbox on the main window
* Contrast inversion
* z_step now defaults to 0.005
* Saving throughput in MB/s in the info manager when recording
* Fix not being able to open large holo files due to an overflow
* Fix a crash / freeze when opening a file with parameters overloading the GPU
* Fix holographic mode not triggering when changing algorithm from None to FFT

### v.7.3

* Toggle reticle on view window
* Scale reticle with "reticle_scale" in Holovibes.ini
* Fix holo file segfault

### v.7.2

* CMake build
* .holo file format
* Bugfixes on Xib camera
* General code cleanup

### v.7.1

* add: dark mode for chart window
* up: replace std::cout calls by logger class calls
* add: logger class
* up: Change chart autoscale offset
* up: Remove qwt library, plot now uses QtCharts
* fix: Use std::filesystem instead of std::experimental::filesystem
* up: Remove complex mode
* up: remove *.ccincludes for moc_* files
* up: remove vibrmetry
* up: remove flowraphy
* up: fix build ad Qt moc file generation
* up: use QGuiAppication for screen geometry
* up: remove stftlongtimes
* fix: .cu fix arument type warning with static_casts
* up: composite.c / hsv.cu remove debug gpu function causing warnings
* fix: array.hh postprocessing.cc fix initialize type warnings
* fix: tools.cu /MainWindow.cc fix initialize type warnings
* up: compute_desriptor.hh fix std::atomic warnings + code format
* up: bump CUDA vrsion to 10

### v.7.0

?

### v.6.7

* fix: plot is now on signal on launch
* fix: hsv h part on convolution

### v.6.6

* add doc stft longtimes
* add: icon

### v.6.5

* up:stft longtimes is now only on magnitude
* fix: imgtype can't change when stft longtimes is checked

### v.6.4

* fix: rainbow hsv is now in stftcuts
* up: pmin and pmax are now copied from rgb to and from hsv
* fix: can't do cropped stft and stft longtimes at the same time
* add: convolution on composite image

### v.6.3

* fix: convolution bug due to auto contrast
* fix: for stft longtimes p and pacc are dependent of #img 2
* fix: camera window size

### v.6.2

* cuda v10
* fix: main window doesn't show stft longtimes
* fix: p maximum is dependent of pacc
* up: convolution kernel is now circshift

### v.6.1.2

* up: fixed frames of xz yz cuts when longtimes
* fix: pacc limit and remove composite in stft longtimes

### v.6.1.1

* add: new mode stft longtimes
* fix: pixel size not updated
* add: first holovibe testsuite

### v.6.1

* up: hsv should be fine now (blur added)

### v.6.0.9

* fix: corrected hue part of HSV
* add: can divide by convolution
* add: renormalize image after convolution
* fix: convolution works
* fix: optimized reading of convolution kernel

### v.6.0.8

* add: HSV representation

### v.6.0.7

* fix: Add of the automatic set of zmin et zmax for the autofocus
* fix: initializing of a file holovibes.ini if it doesn't exist
* add: bfml files for adimec camera
* add: direct and algorithm none are now rectangular
* fix: Change the step for the plot auto scale to avoid strange behaviors

### v.6.0.6

* up: update MainWindow.cc to add documentation
* add: documentation first steps with holovibes
* add: normalization in image rendering panel

## September 2018 - December 2018 ##

Ellena D., Clement F., Danae M. and Hugo V. are now working on the project until December 2018.

### v.6.0.5

* fix: average computation in plot

### v.6.0.3

* xiq: potential fix for error on closing (not tested with the camera)
* direct_refresh: commented the function
* fix: keep scale and translation when changing image type
* updated doxygen

### v.6.0.2

* fix: restored direct refresh function, still needed for direct recording

### v.6.0.1

* add: record synchronized with input file.
* add: negative m for zernike

### v.5.9.0

* change: show all tabs by default except MotionFocus and PostProcessing
* fix: nSize limited to 128 when reading file
* fix: zernike polynomials now correctly called and take into account the factor

### v.5.8.0

* add: possibility to change phase Number during stft cuts
* camera: fixed XIB closing errors
* aberrations: front end
* aberration: fft done using a temporary buffer
* add: cuda error checks after fft c2c
* remove: useless fields of ini file
* gui: changed titles and layouts

### v.5.7.4

* add: zoom lock in reset transform
* add: drag&drop import path

### v.5.7.3

* fix: Now using windows 10 sdk
* add: cuda 9.1

### v.5.7.2

* add: displaying of the output queue in the info tab

### v.5.7.1

* fix: disable whole AutofocusGroupBox instead of just the 'Run Autofocus' PushButton
* add: main doxygen page

### v.5.7.0

* add: rainbow_overlay
* fix: kernel_composite /(red - blue)
* fix: wheel event is only handled in direct window now.
* add: RGB weight add: precompute colors to accelerate XY composite

### v.5.6.7

* add: now resizing to original size when we uncheck square_pixel
* remove: frameless SliceWindow
* add: raw view in hologram.
* add: recording raw while we are in hologram mode
* fix: raw recording available in slices

### v.5.6.6

* change: thread_sync: added a generic way to run a function on mainwindow in the right thread
* fix: slices are now closed properly. Crop stft is now working with stft cuts.
* fix: prevent slice window from moving when resizing
* fix: camera hamamatsu uses the same sdk than the others (8.1)
* fix: resized correctly when passing from composite to direct mode
* add: minimum size to SliceWindow
* add: frameless slice window
* add: resizing slice window automatically remove square_pixel

### v.5.6.4

* fix: autocontrast_slice request added to have autocontrast on both windows at the same time.

### v.5.6.3

* fix: right click doesn't dezoom in slice anymore
* fix: 24-bit recorder is now working properly.

### v.5.6.2

* add: composite_area: front end to select the area
* fix: recording_24: writing the correct bytes
* fix: divided InputThroughput by stft_steps change: 4 decimals in Scale Bar correction SpinBox
* fix: cudav9 used in camera utils

### v.5.5.5

* change: stft steps maximum value set to 2^31
* add: progress bar: bar showing the progress in the input file
* fix: interpolation: added the categori in ini file

### v.5.5.4

* fix: camera: xib: better default values
* add: Hamamatsu camera
* add: stft is now limited to the zoomed area
* fix: now direct mode don't do any computation (issue direct_refresh)
* fix: autofocus and fft/filter2d have now their own classes managing their buffers. lot of code simplifying.
* add: class contrast used to handle the autocontrast part of the pipe
* fix: fft shift and log are now in the contrast class
* add: converts class containing convertion complex to float + that will contain float to short
* fix: free in correct destructors
* fix: composite images fix : stft_env_ passed instead of stft_buf

### v.5.5.3

* fix: memory_allocation: use unique_pointers for some queues
* fix: raii: moving gpu buffers to unique pointers (WIP)
* fix: raii: removed most raw memory management (on CPU side)
* fix: raii: removed some raw pointers
* fix: since we don't disable stft while closing window anymore, we need to check in the pipe::exec if we're in direct mode (we do not want to take steps into account in direct)

### v.5.3.0

* Add: XY stabilization
* Fix: Overlay overhaul
* Add: displaying slice in filename recorded
* Remove: average plot in ini file
* Remove: Selection of the number of bits in records. Now selecting automatically
* Add: XY Zoom in stft view mode
* Add: Now Using Visual Studio 2017, Cuda 9.0 and Boost 1.65.1
* Fix: GUI overhaul
* Fix: Code cleaning

### v.5.2.2

* Fix: PhotonFocus camera bug on first frames

### v.5.2.2

* Add: PhotonFocus camera

### v.5.2.0

* Add: Composite image
* Add: Autofocus reworked, working in STFT
* Add: lens displayed as complex numbers

### v.5.1.1

* Add: FFT shift enable automatically after filter2D selection
* Add: developper names
* Add: Change output type checkboxes into a comboBox

* Remove: average enabled in holovibes.ini
* Remove: 32bits float output type

### v.5.1.0

* Add: black bar in fullscreen mode
* Add: averaging over x, y and p range and displayed by red overlay on each view
* Add: image accumulation for each view (XY, XZ and YZ) independently

* Remove: External trigger

* Fix: 2D filter selection direction was inversed when cursor was not in the bottom-right quadrant
* Fix: Both ',' and '.' are supported as decimal separator

## September 2017 - December 2017 ##

Eloi C., Florian L., and Julien G. are now working on the project until December 2017.

### v.5.0.0

* Update: Qt 5.8 -> 5.9
* Update: STFT is enabled by default
* Add Beta feature: 3D view
* Phase accumulation
* Add: STFT cuts:
    - Record
    - Image type (Modulus / Squared Modulus / Argument / Complex)
* Full screen
* External Trigger (STFT)
* Update: Rotation and Flip mechanism

### v.4.3.0

* Add: STFT cuts :
    - Cross Cursor
    - Auto contrast at start
* Add: FocusIn event to select windows in program
* Add: Direct output buffer

### v.4.2.0

* Add: STFT cuts :
    - Rotation and flip
    - Contrast
    - Log Scale
    - Spacial images accumulations
* Add: title detect for imported files
* Add: New colors for Autofocus, 2DFilter, and Average when selected
* Update: Record now outputs a raw with properties in title readable by title detect feature
* Update: OpenGL now using Dynamic pipeline
* Update: Pop-ups are now displayed in info bar
* Several bugs fixed

## January 2017 -- July 2017 ##

Alexandre B., Thomas J. are now working on the project until July 2017.

### v.4.0.0

* Bug fixing
* Jump a version to match CNRS repository (4.0)


### v.2.5.0

* Bug fixing
* Errors in Holovibes are now displayed & better handled.


### v.2.4.0

* STFT has been rewritten completely with great result improvement
* Thread_reader as been completely rewritten.
* 2D_unwrap has been rewritten.
* Every dll has been updated : Hovovibes is now in 8.0.
* Add Reference substract from the current image (sliding and settled)
* Add Filter2D feature.
* Add Complex display & record


### v.2.3.0

* FFT1 & FFT2 fixed
* FFT1D temporal performances greatly improved
* add accumulation stack feature
* fix several bugs


### v.2.2.0

* Add float file import
* Add complex file write/import
* Add complex color display
* several GUI changes


### v.2.1.0

* Add: demodulation mode
* Add: Special queue to handle Convolution & Flowgraphy
* Add: Convolution with loaded kernels
* Add: temporal Flowgraphy
* Add: cine file parsing
* Add: GUI themes


### v.2.0.1

* Several bugs fixed


## September 2016 -- January 2017 ##

Cyril, Cl�ment are now working on the project until January 2017.


### v.1.5.0

* Camera Adimec can now run up to 300Hz (instead of the previous 60Hz)
* Holovibes can still be launched if visa.dll is not installed as it is now loaded only when needed
* Holovibes can unwrap in 6 different ways the images
* New part in the GUI, displaying information like the framerate
* Dependencies are now handled in a smarter way (not including things where they should not be)
* Ini file is now more dense and can configure more parameters


### v.1.4.0

* Pipeline has been renamed to Pipe, and Pipeline now refers to a another way to treat buffers, with several actions being done at the same time by workers


### v.1.3.0

* GPIB is now working with VISA, handling several instructions per blocks
* Minor changes


### v.1.2.0

* Adimec Quartz A2750 camera is now supported
* Stft can be applied to the image (instead of fft1/fft2)


### v.1.1.0

* The project works on CUDA 7.5
* The project works on QT 5.5
* Edge camera is now heavily handled
* XIQ now supports ROI


### v.1.0.2

* The autofocus can now be done recursively, allowing it to be performed must faster.
* Batch ouput can be done as float.
* Zoom is properly working now.
* The interface is responsive


### v.1.0.1

* Instead of taking the source images directly from the camera, the user can specify an input file (.raw) which will be read by the program.
* The autofocus has been improved a lot in terms of precision.
* Some small performance changes.


## September 2015 -- January 2016 ##

Eric, Arnaud and Geoffrey are now working on the project until January 2016.


### v1.0.0 ###

Major release

* Add documentation `;-)`
* Add HoloVibes' icon


### v0.7.4 ###

* Minor GUI fix


### v0.7.3 ###

* GUI average/ROI plot and record bug fix
* Minor GUI fix


### v0.7.1 ###

* All records can now be stopped
* Autofocus is now handling selection zones
* Minor GUI changes


### v0.7 ###

* Add autofocus algorithm (using global variance and average magnitude - sobel)
* Some minor fixes
* update to qwt 6.1.2
* update to qt 5.4


### v0.6.1 ###

* CSV plot and batch simultaneously


### v.0.6.0 ###

* CSV batch


### v.0.5.9 ###

* Improved DLL loader
* Add smart pointers (memory safety)


### v.0.5.8 ###

* GUI bind gpib
* GUI minor bugs fixed
* Recorder bug fixed
* Camera DLL loader


### v.0.5.7 ###

* GUI ROI plot is freezing when recording
* GUI changing ROI plot number of points on the fly


### v.0.5.6 ###

* GUI plot average can be resized
* GUI auto scale for plot average
* GUI minor bugs fixed


### v.0.5.5 ###

* GUI plot average


### v.0.5.4 ###

* GUI/CLI minor changes
* Update iXon camera


### v.0.5.3 ###

* GUI bug fixes
* Add average values recording


### v.0.5.2 ###

* Fix FFT shift corner algorithm


### v.0.5.1 ###

* Fix dequeue method for Recorder


### v.0.5 ###

* Zone selection (OpenGL)
* Zoom (OpenGL)
* Signal/Noise zone selection average
* Auto loading/saving Holovibes' configuration in INI file
* Minor bug fixes
* New PCO cameras handling (+ pco.edge, camera checking)
* Auto-contrast
* Camera Andor iXon support (beta)


### v.0.4.3 ###

* GUI
  * Visibility methods changed to handle both case instead of having two methods
  * SpinBox sizes changed
  * Names of windows and parameters changed
  * When log scale is enabled, the value given to the pipeline is not powered to 10^9
  * When contrast enabled, enabling the log scale will update the contrast (to convenient values).
* Update CLI
  * Display mode is not available in --nogui mode
  * No Qt in CLI mode


### v.0.4.2 ###

* Add Qt user interface.
  * Keyboard shortcuts
  * New Qt OpenGL window (resizeable)
  * Updates holograms parameters in live
    * Guards and protections against bad user actions
    * Camera change on the fly
    * Record canceling feature
* Add pipeline to apply algorithms.
* Fix queue issue when using big endian camera (Queue ensures that his content is little endian).
* Better memory management (less cudaMalloc), resources are allocated once at start and only few reallocations occurs when tweaking the Phase# parameter.
* Shift corners algorithm has been optimized.
* Make contiguous has been improved (no cudaMalloc and cudaMemcpy).
* Add manual contrast correction.
* Add log10 filter.
* Post-FFT algorithms are more precise (using floats instead of ushorts).
* Thread shared resources are no longer allocated in threads.
* CUDA kernels are less monolithic (complex-real conversions are separated).
* Fix pixelfly hologram mode.
* CLI updated (--nogui mode)
* CLI can set parameters on GUI.


### v.0.3.0 ###

* Add FFT2 algorithm written in CUDA.
* Images to be displayed remains in GPU memory (no more copy to CPU-GPU).
* Fix queue endianness handling.
* 8-bit frames are rescaled in 16-bit frames before computation.


### v.0.2.0 ###

* Add FFT1 algorithm written in CUDA.
* When FFT is enabled, the recorder writes output frames.
* When FFT is enabled, the display shows output frames at 30fps.
* Some bug fixes.


### v.0.1.1 ###

* Fix recorder issue.
* Fix pike 16-bit issue.


### v.0.1 ###

* Command line front-end.
* Handles 4 cameras: xiq, ids, pike, pixelfly.
* Use INI files to configure cameras.
* OpenGL display.
* Records frames.
