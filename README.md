# Holovibes #


Holovibes is a software program that allow to make holographic videos. It is developed in `C++` language.

## User dependencies ##

* Microsoft Windows 10 x64
* Microsoft Windows 7 x64

### Cameras drivers ###

* [XiAPI](http://www.ximea.com/support/wiki/apis/XIMEA_API_Software_Package) XiQ Camera XIMEA API V4.01.80
* [Driver IDS](http://en.ids-imaging.com) V4.41
* [AVT Vimba](https://www.alliedvision.com/fileadmin/content/software/software/Vimba/Vimba_v1.4_Windows.exe) V1.4
* [PCO.Pixelfly USB 2.0 driver](http://www.pco.de/support/interface/sensitive-cameras/pcopixelfly-usb/) V1.04
* [PCO.Edge 4.2 USB 3.0 driver](https://www.pco.de/support/interface/scmos-cameras/pcoedge-42/) V1.08
* Andor iXon SDK *link not available* V2.9
* [Bitflow Cyton CXP4 driver for holovibes] (given with holovibes) Framegrabber for the adimec V6.30
* [PhotonFocus MV1-D1312 driver](http://www.photonfocus.com/en/products/camerafinder/camera/?prid=70) V3.1.14

### VISA drivers ###

* [VISA Drivers](https://www.ni.com/visa/)

### Developpers Setup ###

* Make sure the following installations respect the paths set in "PropertySheet.props":
    * Visual Studio 2017
    * CUDA 9.2 (AFTER installing visual Studio)
    * Qt 5.9.0
    * QT VS TOOLS (Usefull Visual studio add-on)
    * Install and build Qwt 6.1.3 in the "lib" directory
    * Boost 1.65.1 (A prebuilt version is easier. If you build it yourself, make sure libs are built in $(BoostLib)\lib64-msvc-14.1)
    * Doxywizard (optionnal, usefull to manipulate 'doxygen_config' file)
* Make sure the environment variable "CUDA_PATH_V9_1" is set
* Make sure your path contains:
    * $(CUDA_PATH_V9_1)\bin
    * $(CUDA_PATH_V9_1)\libnvvp
    * $(QTDIR)\bin
    * $(QWTDIR)\lib
* After modifying your path, if Holovibes cannot find the Qt platform "windows", redownload Qt.
* Verify that it builds with the correct dll. If not, your path contain something wrong.
    

### IDE ###

* Visual Studio 2017 Community 15.6.7

### CUDA ###

* [CUDA 9.2 Production Release](https://developer.nvidia.com/cuda-downloads)

### Visual C++ redistributable ###

* [Visual C++ Redistributable Packages for Visual Studio 2013](http://www.microsoft.com/en-US/download/details.aspx?id=40784)
* [Visual C++ Redistributable Packages for Visual Studio 2015](https://www.microsoft.com/fr-fr/download/details.aspx?id=48145)

### Qt 5.9.0 ###

* [Qt OpenSource Windows x64 5.9.0](https://download.qt.io/archive/qt/5.9/5.9.0/)


### Libraries ###

* [Boost C++ Library](http://sourceforge.net/projects/boost/files/boost-binaries) 1.65.1
* [Qwt 6.1.3](http://sourceforge.net/projects/qwt/files/qwt/6.1.3/)

## Typical Usage ##

### GUI - *Graphical User Interface*

1. *Configure* the camera (xiq for example) with .ini.
2. Double-click on holovibes.exe icon.

### CLI - *Command Line Interface*

1. *Configure* the camera (xiq for example) with .ini.
2. *Launch* Holovibes using the *Windows PowerShell* for example:

~~~
./Holovibes.exe -c xiq -w 10 frames.raw --1fft -n 2 -p 0 -l 536e-9 -z 1.36
~~~

This enables holograms (FFT1 algorithm) computation using the XiQ camera, recording 10 frames in file `frames.raw`.

#### About CLI mode

This mode is useful to do batch computation (call holovibes from a script).

## For developpers

### Adding a camera  ###

* Right click on "Camera Libbraries"
* Add -> new project -> empty project
* right click on your project -> properties -> change '.exe' to '.dll'
* View -> property Manager
* For both "Debug | x64" and "Release | x64":
    * Add -> Add existing property sheet -> "PropertySheet.props" and "CameraDLL.props"
* Make sure "CAMERA_EXPORTS" is set (properties -> C/C++ -> preprocessor -> Preprocessor definitions)
* right click on Holovibes -> Build Events -> Post-Build Events -> add the copy of your SDk dll and your ini file
* Do not forget updating setupCreator.iss to copy your newly created Dll and .ini file

### Linker Errors ###

Cuda functions

* Go to: Project>Holovibes property>Linker>Input : Add the missing .lib (example: "nppc.lib" )
* Go to: Project>Holovibes property>C/C++>Command Line>Additionnal options : Add missing "-l*" (example: "-lnppi_static")

## Features ##

* Command line options (for batch processing only)
* Works with these cameras:
    * PCO Edge 4.2 LT
    * Andor iXon +885
    * IDS imaging CMOSIS CMV 4000 
    * AVT Pike Kodak KAI 4022 F-421 
    * PCO Pixelfly
    * Ximea CMOSIS XiQ
    * Ximea CMOSIS XiB
    * Adimec Quartz-2A750 (with Bitflow Cyton-CXP4 framegrabber)
    * PhotonFocus MV1-D1312IE-100-G2-12
    * Hamamatsu C11440

* Cameras configuration using INI files
* OpenGL realtime display
  * Zone selection (zoom, signal/noise averaging, filtering, autofocus)
* Record frames
* Hologram computation using the FFT1/FFT2/STFT algorithms
* Logarithm filter
* Manual contrast (logarithmic scale values)
* Auto-contrast
* Various view modes : *magnitude*, *squared magnitude*, *argument*, *unwrapped phase*
* Graphical user interface
  * Change parameters on the fly
  * Shortcuts for convenience
* Vibrometry analysis
* Choice between two modes of computation : sequential and parallel
* Settings are automatically saved upon closure of the program
* Batch savings (CSV)
* Batch output as raw images, controlling instruments with GPIB interface
* Autofocus
* Average plot real-time display
* Importing .raw images instead of taking the input from the camera
* Various runtime informations on the program's state
* Easy to use installer for the software
* Image stabilization
* Image coloring
* averaging in the three axis
* lens displaying

## Authors ##

* Ellena Davoine
* Clement Fang
* Danae Marmai
* Hugo Verjus
* Eloi Charpentier
* Julien Gautier
* Florian Lapeyre
* Thomas Jarrossay
* Alexandre Bartz
* Cyril Cetre
* Clement Ledant
* Eric Delanghe
* Arnaud Gaillard
* Geoffrey Le Gourrierec
* Jeffrey Bencteux
* Thomas Kostas
* Pierre Pagnoux
* Antoine Dillée
* Romain Cancillière
* Michael Atlan

## Git ##

If you do not know how to use git, please have a look at the following tutorial.
	* Git - Documentation
	* Bitbucket Git FAQ

### Clone the repository

You can use a GUI tool as SourceTree or use git in command line.
git clone https://username@bitbucket.org/PierrePagnoux/holovibes.git

### Git rules

To let the versioning tool consistent, you have to respect these rules.
* master branch must be clean and compile.
* Never push generated files.
* Use branch for everything. For example to develop a new feature : new/myfeature.
*  Prefer use rebase when pulling changes in your own branch (it avoids merge commits).
* Use merge when pushing your changes to another branch.
* Never commits on master branch directly (without the acknowledge of your team mates).
* Commit messages: use keywords as ‘add, fix, rm, up, change’
* Git rules - Code review
* Git names conventions

## Changelog ##

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


# Known problems :

* Marshall.cxx / ole32.dll : No known fixes, try updating QT.
* Camera not recognized in holovibes : Make sure your versions of .dll and .lib files are updated.
* If the holograms do not work for a reason, try checking what gpu architecture you are compiling on (20, 30, 35...).
