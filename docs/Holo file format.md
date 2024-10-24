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

### 64 bytes binary header

- "HOLO" magic number (4 bytes)
- Version of the Holo file format (Latest: 7)
- Number of bits per pixel (2 bytes)
- Width of the images (4 bytes)
- Height of the images (4 bytes)
- Number of images (4 bytes)
- Total data size (8 bytes)
- Endianness (1 byte)
- Data type (1 byte)

### Image data

The raw image data

### Json footer

The footer has the same format as saved Compute Settings, located in Appdata.
The four root fields correspond to main UI panels; they are:

- advanced
- composite
- image_rendering
- view

## Implementation

The implementation for the holo file format can be found in:

- Holovibes/sources/io_files/holo_file.cc
- Holovibes/includes/io_files/holo_file.hh

The format of the Compute Settings can be found in:

- Holovibes/sources/compute_settings.cc
- Holovibes/includes/struct/compute_settings_struct.hh

Last updated: 24/10/2024
