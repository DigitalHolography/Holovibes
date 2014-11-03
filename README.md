# Holovibes v0.1.1 #

Holovibes is a software program that allow to make holographic videos. It is developed in `C++` language.

## User dependencies ##

### Cameras drivers ###

* [XiAPI](http://www.ximea.com/support/wiki/apis/XiAPI) XiQ Camera XIMEA API V4.01.80
* [Driver IDS](http://en.ids-imaging.com) V4.41
* [AVT Vimba](http://www.alliedvisiontec.com/us/products/legacy.html) V1.3
* [PCO.Pixelfly driver USB 2.0](http://www.pco.de/support/interface/sensitive-cameras/pcopixelfly-usb/) V1.04

### CUDA ###

* [CUDA 6.5 Production Release](https://developer.nvidia.com/cuda-downloads)

### Visual C++ redistributable ###

* [Visual C++ Redistributable Packages for Visual Studio 2013](http://www.microsoft.com/en-US/download/details.aspx?id=40784)

## Developers dependencies ##

### Libraries ###

* [Boost C++ Library](http://sourceforge.net/projects/boost/files/boost-binaries) 1.55.0 build2

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

## Authors ##

* Michael ATLAN <michael.atlan@espci.fr>
* Jeffrey BENCTEUX <jeffrey.bencteux@espci.fr>
* Thomas KOSTAS <thomas.kostas@espci.fr>
* Pierre PAGNOUX <pierre.pagnoux@epita.fr>

## Changelog ##

### v.0.1.1 ###

* Fix recorder issue.
* Fix pike 16-bits issue.
