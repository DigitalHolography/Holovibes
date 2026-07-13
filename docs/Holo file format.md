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
- Version of the Holo file format (2 bytes)
- Number of bits per pixel (2 bytes)
- Width of the images (4 bytes)
- Height of the images (4 bytes)
- Number of images (4 bytes)
- Total data size (8 bytes)
- Endianness (1 byte)
- Data type (1 byte)

- Padding (34 bytes)

### Image data

The raw image data

### JSON footer

All bytes after the image data form one JSON document. Its current top-level
shape is:

```json
{
  "compute_settings": {},
  "info": {}
}
```

`compute_settings` has the same format as the saved Compute Settings located in
AppData. It contains `version`, `advanced`, `color_composite_image`,
`image_rendering`, and `view`. `info` contains properties of the recording such
as pixel pitch, input FPS, camera information, contiguity, and timestamps.

#### Timestamps

`info.timestamps_us` always contains the session bounds and duration in
microseconds:

```json
{
  "unix_first": 1000000,
  "unix_last": 1000250,
  "duration": 250,
  "camera_first": 100000,
  "camera_last": 100249,
  "offset_first": 900000,
  "offset_last": 900001
}
```

When **Per-frame timestamps** is enabled for a RAW camera recording, an
optional `per_frame` object is added:

```json
{
  "per_frame": {
    "unix": [1000000, 1000125, 1000250],
    "camera": [100000, 100125, 100249],
    "offset": [900000, 900000, 900001]
  }
}
```

The three arrays have the same length. Index `i` in each array describes image
`i` in the file, after record offset and frame skipping have been applied.
`unix` is the host-synchronized value supplied to the pipeline, `camera` is the
camera clock value, and `offset` is the camera-to-host clock offset. Backends
that do not provide timestamp metadata leave the corresponding values at `0`.

Per-frame timestamp capture currently applies only to RAW recordings made
directly from a camera. A camera API may deliver only one hardware timestamp for
a batch of frames; in that case Holovibes derives the other entries from the
batch's first timestamp and nominal frame period. The arrays therefore describe
the timestamp information available to the recording pipeline and are not
necessarily independent hardware measurements for every frame.

## Implementation

The implementation for the holo file format can be found in:

- `Backend/includes/io/holo_file.hh`
- `Backend/sources/io/input_file/input_holo_file.cc`
- `Backend/sources/io/output_file/output_holo_file.cc`

The format of the Compute Settings can be found in:

- `Backend/sources/core/compute_settings.cc`
- `Backend/includes/struct/compute_settings_struct.hh`

Last updated: 13/07/2026
