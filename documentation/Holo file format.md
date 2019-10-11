# Holo file format

## General infos

The holo file format is designed to save some metadata about the raw images so
that the users don't have to reset their settings every time they import a file.
It is composed of a header, the image data and a footer. It was designed this
way so that it could be easily openend in ImageJ (with a fixed offset from the
beginning) and more meta data could be added to it without changing the size
of the header. Holovibes is still retrocompatible with the old naming convention
of .raw files.

## File content

The holo file format is designed this way:

### 18 bytes binary header

* "HOLO" magic number (4 bytes)
* Number of bits per pixel (2 bytes)
* Width of the images (4 bytes)
* Height of the images (4 bytes)
* Number of images (4 bytes)

### Image data

The raw image data

### Json footer

The fields names are (almost) the same as the ones on the Holovibes frontend

* algorithm
* #img
* p
* lambda
* z
* log_scale
* contrast_min
* contrast_max
* endianess

The following fields are here to make it easier to create holo files even though
they are already present in the header

* img_width
* img_height
* pixel_bits

## Implementation

The implementation for the holo file format can be found in:

* Holovibes/sources/holo_file.cc
* Holovibes/includes/holo_file.hh

Last updated: 11/10/2019
