# Laser speckle contrast imaging

Select **Processed** image mode, then **LSCI** in the time transformation
selector. Supply direct single-channel speckle intensity images acquired without
a reference beam. Supported input formats are 8-bit, 16-bit, and float32.
LSCI sets Space Transform to **None** and disables reconstruction and focus
controls.

LSCI follows the estimator described in *High-speed laser speckle contrast
imaging of the human retina*, Auray et al., October 25, 2026, section II.D:

```text
mean = sum(I) / samples
variance = max(sum(I * I) / samples - mean * mean, 0)
K = sqrt(variance) / mean       (zero when mean <= 0)
```

The samples pool a centered S x S spatial neighborhood across N consecutive
frames. This is one population variance over the entire neighborhood, including
both spatial and temporal variation. Neighborhoods are clipped at image borders;
the denominator uses the actual number of contributing pixels times N. All
statistics use 32-bit floating point, with the usual cancellation limits near
constant signals. No noise, motion, dark-offset, or illumination correction is
applied.

The default spatial side length S is **5** pixels, so the neighborhood is
**5 x 5 pixels**. Change **LSCI space window** to choose an odd side length from
**1 to 63** pixels. A value of 1 uses temporal statistics at each pixel without
pooling adjacent pixels. The separate **Time window** control sets N, the number
of consecutive frames; choosing LSCI sets N to **16**. Use S = 5 and N = 16 to
match the paper's estimator. The spatial window is saved in compute settings and
restored when those settings are loaded.
Larger neighborhoods reduce spatial detail and require more processing.

Blocks do not overlap, and an incomplete block does not produce a map. A queue
reset or a change to the spatial window, temporal window, or batch size discards
a partial block. An input-queue overflow also invalidates the block and discards
the backlog so the next map starts with fresh consecutive input. Active frame
recording ends with the existing overflow indication.

Every received input batch enters the estimator. Batch size must divide N and
is adjusted down to a valid divisor when necessary. Time stride is fixed to N
and represents the output interval in LSCI mode. Thus, 1000 input frames/s with
N = 16 gives a nominal 62.5 contrast maps/s, independently of batch size. This
is a cadence calculation, not a throughput or latency guarantee.

The implementation accumulates per-pixel intensity sums and squared sums, then
pools them spatially. Its statistics buffer occupies one float2 image regardless
of N or S, without allocating temporal complex cubes or reconstruction buffers.
Spatial reconstruction and temporal Fourier transformation are bypassed.

Input filtering, filter and lens views, phase/component selection, frequency
cuts, moments, and OCT exports are unavailable in LSCI. Convolution,
renormalization, registration, and extra image averaging are bypassed. Rotation,
flip, display contrast, and logarithmic display remain available. Entering or
leaving LSCI restarts processing. Stop recording before changing the algorithm,
spatial window, temporal window, or batch size.

Recording options are:

- **Raw:** the original direct intensity frames.
- **Contrast map:** rendered 16-bit images, including display settings. Before
  display processing, K is multiplied by 65535. With contrast and logarithm
  disabled, this gives a fixed display scale with values above one clipped.
- **Chart:** the existing regional statistics, calculated from dimensionless K
  before display scaling. To reproduce the paper's regional inverse contrast,
  take the reciprocal of the regional average K after export. Averaging 1/K
  instead gives a different quantity.

Full floating-point K image export is not included. Contrast maps are relative
speckle observables; no conversion to calibrated flow velocity is performed.

For an existing compute-settings JSON, the relevant fields are:

```json
{
  "version": "v1",
  "image_rendering": {
    "time_transformation": "LSCI",
    "lsci_spatial_window": 5,
    "time_transformation_size": 16,
    "time_transformation_stride": 16,
    "batch_size": 4,
    "space_transformation": "NONE"
  }
}
```

Loaded LSCI settings normalize incompatible processing options automatically.
The `lsci_spatial_window` field is the odd side length in pixels, not the number
of pixels in the square. Settings files without this field use the default 5.
