# Holovibes v0.5.7 #

Holovibes is a software program that allow to make holographic videos. It is developed in `C++` language.

## User dependencies ##

* Microsoft Windows 7 x64

### Cameras drivers ###

* [XiAPI](http://www.ximea.com/support/wiki/apis/XiAPI) XiQ Camera XIMEA API V4.01.80
* [Driver IDS](http://en.ids-imaging.com) V4.41
* [AVT Vimba](http://www.alliedvisiontec.com/us/products/legacy.html) V1.3
* [PCO.Pixelfly USB 2.0 driver](http://www.pco.de/support/interface/sensitive-cameras/pcopixelfly-usb/) V1.04
* [PCO.Edge 4.2 USB 3.0 driver](http://www.pco.de/support/interface/scmos-cameras/pcoedge-42/) V1.08

### CUDA ###

* [CUDA 6.5 Production Release](https://developer.nvidia.com/cuda-downloads)

### Visual C++ redistributable ###

* [Visual C++ Redistributable Packages for Visual Studio 2013](http://www.microsoft.com/en-US/download/details.aspx?id=40784)

### Qt 5.3.2 ###

* [Qt OpenSource Windows x64 MSVC OpenGL 5.3.2](http://download.qt-project.org/official_releases/qt/5.3/5.3.2/qt-opensource-windows-x86-msvc2013_64_opengl-5.3.2.exe.mirrorlist)

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

* [Boost C++ Library](http://sourceforge.net/projects/boost/files/boost-binaries) 1.55.0 build2

### IDE ###

* Visual Studio 2013 Professional

## Features ##

* Command line options (for batch processing only)
* Works with 6 cameras:
    * Edge
    * iXon
    * IDS
    * Pike
    * Pixelfly
    * XiQ
* Cameras configuration using INI files
* OpenGL realtime display
  * Zone selection (zoom, signal/noise averaging)
* Record frames
* Hologram computation using the FFT1/FFT2 algorithms.
* Logarithm filter
* Manual contrast (logarithmic scale values)
* Auto-contrast
* Three view modes : *magnitude*, *squared magnitude* and *argument*.
* Graphical user interface
  * Change parameters on the fly
  * Shortcuts for convenience
* Vibrometry analysis.
* Settings auto saving.

## Authors ##

* Michael ATLAN <michael.atlan@espci.fr>
* Jeffrey BENCTEUX <jeffrey.bencteux@epita.fr>
* Thomas KOSTAS <thomas.kostas@epita.fr>
* Pierre PAGNOUX <pierre.pagnoux@epita.fr>

## Changelog ##

### v.0.1 ###

* Command line front-end.
* Handles 4 cameras: xiq, ids, pike, pixelfly.
* Use INI files to configure cameras.
* OpenGL display.
* Records frames.

### v.0.1.1 ###

* Fix recorder issue.
* Fix pike 16-bit issue.

### v.0.2.0 ###

* Add FFT1 algorithm written in CUDA.
* When FFT is enabled, the recorder writes output frames.
* When FFT is enabled, the display shows output frames at 30fps.
* Some bug fixes.

### v.0.3.0 ###

* Add FFT2 algorithm written in CUDA.
* Images to be displayed remains in GPU memory (no more copy to CPU-GPU).
* Fix queue endianness handling.
* 8-bit frames are rescaled in 16-bit frames before computation.

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

### v.0.5 ###

* Zone selection (OpenGL)
* Zoom (OpenGL)
* Signal/Noise zone selection average
* Auto loading/saving Holovibes' configuration in INI file
* Minor bug fixes
* New PCO cameras handling (+ pco.edge, camera checking)
* Auto-contrast
* Camera Andor iXon support (beta)

### v.0.5.1 ###

* Fix dequeue method for Recorder

### v.0.5.2 ###

* Fix FFT shift corner algorithm

### v.0.5.3 ###

* GUI bug fixes
* Add average values recording

### v.0.5.4 ###

* GUI/CLI minor changes
* Update iXon camera

### v.0.5.5 ###

* GUI plot average

### v.0.5.6 ###

* GUI plot average can be resized
* GUI auto scale for plot average
* GUI minor bugs fixed

### v.0.5.7 ###

* GUI ROI plot is freezing when recording
* GUI changing ROI plot number of points on the fly

### v.0.5.8 ###

* GUI bind gpib
* GUI minor bugs fixed
* Recorder bug fixed
* Camera DLL loader
