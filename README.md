# Holovibes #

Holovibes is a software program that allow to make holographic videos. It is developed in `C++` language.

## Features

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

## Usage

### GUI - *Graphical User Interface*

1. *Configure* the camera (xiq for example) with .ini.
2. Double-click on holovibes.exe icon.

### CLI - *Command Line Interface*

**This mode is useful to do batch computation (call holovibes from a script).**

1. *Configure* the camera (xiq for example) with .ini.
2. *Launch* Holovibes using the *Windows PowerShell* for example:

~~~
./Holovibes.exe -c xiq -w 10 frames.raw --1fft -n 2 -p 0 -l 536e-9 -z 1.36
~~~

This enables holograms (FFT1 algorithm) computation using the XiQ camera, recording 10 frames in file `frames.raw`.


## Dependencies

* Microsoft Windows 7 x64 (or later)

### Cameras drivers

* [XiAPI](http://www.ximea.com/support/wiki/apis/XIMEA_API_Software_Package) XiQ Camera XIMEA API V4.01.80
* [Driver IDS](http://en.ids-imaging.com) V4.41
* [AVT Vimba](https://www.alliedvision.com/fileadmin/content/software/software/Vimba/Vimba_v1.4_Windows.exe) V1.4
* [PCO.Pixelfly USB 2.0 driver](http://www.pco.de/support/interface/sensitive-cameras/pcopixelfly-usb/) V1.04
* [PCO.Edge 4.2 USB 3.0 driver](https://www.pco.de/support/interface/scmos-cameras/pcoedge-42/) V1.08
* Andor iXon SDK *link not available* V2.9
* [Bitflow Cyton CXP4 driver for holovibes] (given with holovibes) Framegrabber for the adimec V6.30
* [PhotonFocus MV1-D1312 driver](http://www.photonfocus.com/en/products/camerafinder/camera/?prid=70) V3.1.14

### VISA drivers

* [VISA Drivers](https://www.ni.com/visa/)