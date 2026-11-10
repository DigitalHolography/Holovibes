# Laser speckle contrast imaging

Select **Processed** image mode, then **LSCI** in the time transformation
selector. Supported single-channel input formats are 8-bit, 16-bit, and float32.
Use **Space Transform = None** for direct speckle
intensity images acquired without a reference beam. To reconstruct the input
before computing contrast, choose **Fresnel Tr.** or **Angular SP.** and adjust
the wavelength and focus controls. In that case, LSCI uses reconstructed
intensity, the squared modulus of the complex field. Saved space-transform
choices are preserved when loading settings or restarting processing.

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
**5 x 5 pixels**. Change **LSCI window size** to choose an odd side length from
**1 to 63** pixels; the control displays the full square size, for example
**5 x 5 px**. A value of 1 uses temporal statistics at each pixel without
pooling adjacent pixels. The separate **Time window** control sets N, the number
of consecutive frames; choosing LSCI preserves the configured N. Use S = 5 and N = 16 to
match the paper's estimator. The spatial window is saved in compute settings and
restored when those settings are loaded.
Larger neighborhoods reduce spatial detail and require more processing.

**Time stride** sets the number of input frames between the starts of successive
windows. A stride below the time-window size overlaps windows; an equal stride
uses consecutive blocks; a larger stride leaves gaps. The first map requires a
complete time window, and an incomplete trailing window produces no map. A queue
reset or a change to the spatial window, temporal window, stride, batch size, or
reconstruction settings discards a partial block. An input-queue overflow also
invalidates the block and discards the backlog so the next map starts with fresh
consecutive input. Active frame recording ends with the existing overflow
indication.

Batch size must divide N, and time stride must be a positive multiple of batch
size. Thus, 1000 input frames/s with N = 16 and stride 4 gives a nominal 250
contrast maps/s after the first 16 frames. This
is a cadence calculation, not a throughput or latency guarantee.

The implementation accumulates per-pixel intensity sums and squared sums, then
pools them spatially. Consecutive or gapped windows use one float2 image for
statistics. Overlap additionally retains N / batch-size float2 images containing
per-batch statistics, without allocating temporal complex cubes. With **Space Transform**
set to **None**, reconstruction buffers are also unnecessary. Selecting a space
transform allocates reconstruction buffers for the current batch. Temporal
Fourier transformation is bypassed in both cases.
A fixed normalization of the Fresnel field avoids floating-point overflow
without changing the dimensionless contrast K.

Input filtering, filter and lens views, phase/component selection, frequency
cuts, moments, and OCT exports are unavailable in LSCI. Convolution and
renormalization are bypassed. **Img acc.** averages the most recent contrast
maps after registration and before regional statistics and display processing.
Level 1 disables averaging; startup uses the maps available so far. Registration
aligns contrast maps; it does not
correct motion within the temporal window. Gaussian smoothing is used only to
estimate motion; the displayed contrast map retains its original detail.
The registration circle shows the central matching region, leaving a margin for motion search.
Rotation, flip, display contrast, and logarithmic display remain available.
Entering or leaving LSCI restarts processing. Stop recording before changing
the algorithm, spatial window, temporal window, stride, batch size, or reconstruction
settings.

Recording options are:

- **Raw:** the original camera frames.
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

## Batch AVI export

Run `ProcessHoloFiles.ps1` and choose configuration mode **3: LSCI AVI**, or
choose **Interactive setup** followed by recording mode **5: LSCI AVI**.
For an unattended export:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ProcessHoloFiles.ps1 -Lsci -InputFolder 'D:\lsci'
```

Use `-OutputFolder` to write the videos elsewhere. The LSCI mode uses the
bundled `settingsLSCI.json` preset: direct intensity, a 5 x 5 spatial window,
16-frame temporal blocks, automatic display contrast, and registration, which
is always enabled in this export mode. Display contrast is estimated once from
the first complete contrast map, using the central 55% circle and the 1st and
99th percentiles to avoid noisy edges. The same display limits are then used
throughout that AVI to prevent brightness jumps; each input file is calibrated
independently. The script passes `--contrast_once` to Holovibes for AVI image
exports. The entire image is exported. The mode retains the input resolution
and records every complete contrast map without additional frame skipping or
accumulation startup skips.

AVI playback is limited to **24 frames/s**, retaining every contrast map. For
4096 input frames acquired at 1000 frames/s, the result contains **256 maps**
and plays for approximately **10.67 seconds** at 24 frames/s. This slows playback
relative to the 4.096-second acquisition. AVI uses MJPEG compression, so its
file size depends on image content; resolution alone does not determine the size.

## Offline PCA-filtered LSCI (CLI only)

Choose configuration **4: Offline PCA-LSCI AVI** in `ProcessHoloFiles.ps1`, or
recording mode **6** in its interactive setup. For an unattended export:

```powershell
.\ProcessHoloFiles.ps1 -LsciPca -InputFolder 'D:\lsci' -OutputFolder 'D:\lsci\pca'
```

This separate offline path implements Surkov et al., *Laser speckle contrast
imaging with principal component and entropy analysis* (2025), sections 2.3,
2.6–2.7 and Figure 2: https://doi.org/10.1007/s12200-024-00143-1.
It is unavailable in the GUI and cannot be activated by a saved compute setting.
It processes direct camera intensities; the existing holographic complex-field
PCA time transformation is a different algorithm.

Each block has **100 raw frames** by default in the direct CLI. Flattened frames
form the columns of X; subtract each frame's spatial mean to obtain Phi. Compute
the small temporal Gram matrix Phi^T Phi and retain the eigenvectors whose
eigenvalues are at least the mean of the N eigenvalues. Reconstruct the static
signal as `X_static = means + Phi V_static V_static^T`. Calculate spatiotemporal
K separately on X and X_static using a **7 x 7** spatial neighborhood, then
subtract **K_original - K_static**. This follows the supplied paper's contrast-map
subtraction, rather than LSCI of a raw residual or the squared-contrast subtraction
in the earlier Arias-Cruz method. There is no conversion to calibrated flow speed.

The implementation calculates the reconstructed stack's temporal moments from
orthonormal PCA scores, giving the same pooled contrast without storing a second
full reconstructed movie. The raw intensity block is held as float32 on the GPU;
the eigendecomposition is only N x N on the CPU. GPU storage is approximately
`4 * width * height * (N + 8)` bytes, plus a small tiled projection workspace.
A 3840 x 2160 x 100 block needs approximately 3.4 GiB, in addition to CUDA/BLAS
overhead. Insufficient memory produces an error suggesting a smaller block.
Reading, processing and writing are synchronous, so slow processing does not
drop input frames or compete with live acquisition.

Inter-map registration is enabled by default. Signed contrast differences are
preserved until display; the 1st and 99th percentiles of the first registered
map's central 55% circle set the display limits for the entire AVI. No logarithm,
extra normalization, reconstruction or per-frame contrast refresh is applied.
Optional image accumulation averages registered, signed maps before display
scaling. A sidecar `<output>.avi.json` records those signed limits, parameters,
the static PC count and eigenvalues for each block, and registration shifts.
AVI remains an 8-bit, lossy MJPEG display product.

Outputs are named `<input>_lsci_pca.avi`. Every complete block produces one map.
Time stride defaults to the block size, preserving consecutive nonoverlapping
blocks. Thus 4096 raw frames
produce **40 maps** at the default block size, with 96 trailing frames unused.
At 24 fps this is 1.67 seconds of playback, independently of acquisition time.

Use `-PcaFrames` and `-PcaWindow` to override the paper defaults, for example:

```powershell
.\ProcessHoloFiles.ps1 -LsciPca -PcaFrames 64 -PcaWindow 5 -InputFolder 'D:\lsci'
```

Set **`-PcaStride`** to any positive number of input frames between block starts.
It defaults to `-PcaFrames`: a smaller stride overlaps blocks and a larger stride
leaves gaps. For example, 32-frame windows starting every 8 frames:

```powershell
.\ProcessHoloFiles.ps1 -LsciPca -PcaFrames 32 -PcaStride 8 -PcaWindow 7 -InputFolder 'D:\lsci'
```

The direct CLI option is **`--lsci_pca_stride 8`**. For F selected input frames,
window size N and stride S, the available map count is `1 + floor((F - N) / S)`
when F >= N. Only complete windows are processed, within the selected `-s`/`-e`
range; `-n` can limit their number. The sidecar records `time_stride` and each
block's first input frame. PCA is recomputed independently for every window, so
overlap increases the amount of processing.

Set `-PcaImageAcc` to average the most recent output maps, for example:

```powershell
.\ProcessHoloFiles.ps1 -LsciPca -PcaFrames 32 -PcaWindow 7 -PcaImageAcc 4 -InputFolder 'D:\lsci'
```

The default is 1 (no averaging); accepted values are 1 to 512. This is a rolling
mean after inter-map registration and before display scaling. Startup averages
only the maps available so far: with level 4, the first frame uses one map,
the second uses two, the third uses three, and subsequent frames use the latest
four. Every complete PCA block still produces one AVI frame. PCA block size,
raw-frame stride and component selection are independent of image accumulation;
the output averaging windows overlap. With 32 raw frames/block and ImgAcc 4,
a full rolling average at the default stride of 32 spans four maps covering 128 raw frames.
Display limits remain fixed from the first output map. The sidecar records the
accumulation level and number of contributing maps for each output frame.
Additional host memory is approximately `4 * width * height * A` bytes for map
history plus `8 * width * height` bytes for the running sum when A > 1.

Direct CLI equivalent:

```powershell
Holovibes.exe -i input.holo -o output_lsci_pca.avi --lsci_pca --lsci_pca_frames 100 --lsci_pca_window 7
```

The new mode accepts raw 8-bit, 16-bit and float32 `.holo` input with even width
and height for MJPEG encoding. `-s` and `-e`
select a 1-based inclusive raw-frame range; `-n` limits the number of output
blocks. `--mp4_fps` sets AVI playback from 1 to 24 fps, and
`--lsci_pca_no_registration` disables inter-map alignment. Custom `-c` compute
settings, frame skipping and raw/moments/OCT recording flags are rejected.
`-f` records the acquisition frame rate in the sidecar; offline processing runs
as fast as the hardware allows, without real-time pacing.
Use `--lsci_pca_image_acc 4` for the direct CLI equivalent of `-PcaImageAcc 4`.
