# Holovibes v4.2.170330 #


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
* [Bitflow Cyton CXP4 driver](http://www.bitflow.com/downloads/bfsdk620.zip) Framegrabber for the adimec V6.20

### CUDA ###

* [CUDA 8.0.61 Production Release](https://developer.nvidia.com/cuda-downloads)

### Visual C++ redistributable ###

* [Visual C++ Redistributable Packages for Visual Studio 2013](http://www.microsoft.com/en-US/download/details.aspx?id=40784)
* [Visual C++ Redistributable Packages for Visual Studio 2015](https://www.microsoft.com/fr-fr/download/details.aspx?id=48145)

### Qt 5.8.0 ###

* [Qt OpenSource Windows x64 5.8.0](https://download.qt.io/archive/qt/5.8/5.8.0/)

### VISA ###

* [VISA Drivers](https://www.ni.com/visa/)

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

## Developers dependencies ##

### Libraries ###

* [Boost C++ Library](http://sourceforge.net/projects/boost/files/boost-binaries) 1.63.0
* [Qwt 6.1.3](http://sourceforge.net/projects/qwt/files/qwt/6.1.3/)

### IDE ###

* Visual Studio 2015 Community

## Features ##

* Command line options (for batch processing only)
* Works with 7 cameras:
    * PCO Edge 4.2 LT
    * Andor iXon +885
    * IDS imaging CMOSIS CMV 4000 
    * AVT Pike Kodak KAI 4022 F-421 
    * PCO Pixelfly
    * Ximea CMOSIS CMV 4000
    * Adimec Quartz-A2750 (with Bitflow Cyton-CXP4 framegrabber)
* Cameras configuration using INI files
* OpenGL realtime display
  * Zone selection (zoom, signal/noise averaging)
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

## Authors ##

* Michael ATLAN <michael.atlan@espci.fr>
* Jeffrey BENCTEUX <jeffrey.bencteux@epita.fr>
* Thomas KOSTAS <thomas.kostas@epita.fr>
* Pierre PAGNOUX <pierre.pagnoux@epita.fr>
* Eric Delanghe <edelangh@student.42.fr>
* Arnaud GAILLARD <arnaud.gaillard@epita.fr>
* Geoffrey LEGOURRIEREC <geoffrey.legourrierec@epita.fr>
* Antoine Dillée <antoined_78@hotmail.fr>
* Romain Cancilliere <romain.cancilliere@gmail.com>
* Clement Ledant <clement.ledant@gmail.com>
* Cyril CETRE <cyril.cetre@epita.fr>
* Thomas JARROSSAY <tjarross@student.42.fr>
* Alexandre BARTZ <abartz@student.42.fr>

## Changelog ##

### v.4.2

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

## January 2016 -- July 2017 ##

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

Cyril, Clément are now working on the project until January 2017.


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