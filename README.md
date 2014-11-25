# Holovibes v0.3 #

Holovibes is a software program that allow to make holographic videos. It is developed in `C++` language.

## User dependencies ##

* Microsoft Windows 7 x64

### Cameras drivers ###

* [XiAPI](http://www.ximea.com/support/wiki/apis/XiAPI) XiQ Camera XIMEA API V4.01.80
* [Driver IDS](http://en.ids-imaging.com) V4.41
* [AVT Vimba](http://www.alliedvisiontec.com/us/products/legacy.html) V1.3
* [PCO.Pixelfly driver USB 2.0](http://www.pco.de/support/interface/sensitive-cameras/pcopixelfly-usb/) V1.04

### CUDA ###

* [CUDA 6.5 Production Release](https://developer.nvidia.com/cuda-downloads)

### Visual C++ redistributable ###

* [Visual C++ Redistributable Packages for Visual Studio 2013](http://www.microsoft.com/en-US/download/details.aspx?id=40784)

## Typical Usage ##

1. *Configure* the camera (xiq for example) with .ini.
2. *Launch* Holovibes using the *Windows PowerShell* for example:

~~~
./Holovibes.exe -c xiq -d 720 --1fft -n 2 -p 0 -l 536e-9 -z 1.36
~~~

This enables holograms (FFT1 algorithm) computation using the XiQ camera, displaying in a square windows of 720x720 pixels.

## Developers dependencies ##

### Libraries ###

* [Boost C++ Library](http://sourceforge.net/projects/boost/files/boost-binaries) 1.55.0 build2
* [GLEW: OpenGL Extension Wrangler Library](http://glew.sourceforge.net)

### IDE ###

* Visual Studio 2013 Professional

## Features ##

* Command line options
* Works with 4 cameras:
    * XiQ
    * IDS
    * Pike
    * Pixelfly
* Cameras configuration using INI files
* OpenGL realtime display
* Record frames
* Hologram computation using the FFT1/FFT2 algorithms.

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
