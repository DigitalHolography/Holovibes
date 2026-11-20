# Local PCA

Select **Local PCA** in the existing Time Transform menu. It uses the usual spatial
reconstruction and image display controls, plus five settings shown only for this
transform:

| Setting | Meaning | Default |
| --- | --- | --- |
| `pca_width_x` | Spatial block width, in pixels | 32 |
| `pca_width_y` | Spatial block height, in pixels | 32 |
| `pca_stride_x` | Horizontal distance between block origins | 16 |
| `pca_stride_y` | Vertical distance between block origins | 16 |
| `pca_rank_factor` | Nonnegative effective-rank multiplier | 1 |

The **Rank factor** slider and numeric field adjust how many strong components
are removed. The multiplier is applied to the unrounded effective rank: below 1
removes fewer components, above 1 removes more, and 0 disables component removal.
The slider covers 0 to 10. The numeric field also accepts larger finite float
values; when the factor exceeds 10, the slider stays at its maximum while the
numeric field preserves the entered value.
One factor is used for every spatial tile in a processed temporal block. Adjusting
it while viewing takes effect on subsequent blocks without reallocating tile buffers.

Block sides and strides must be positive. Each stride must be no larger than its
block side, so the blocks cover every pixel. Reducing a side in the UI also reduces
its stride when necessary.

The shared **Time window** sets the number `T` of consecutive frames per block.
There is no temporal overlap or frame decimation: time stride is fixed to `T`, and
batch size is adjusted to a divisor of `T`. Incomplete temporal blocks produce no
output. The shared spatial transform still runs before Local PCA.

For each spatial origin `(i * pca_stride_x, j * pca_stride_y)` inside the frame:

1. Gather a `pca_width_x * pca_width_y` by `T` matrix `H` of complex samples.
   Samples crossing an image border wrap to the opposite border, including corners.
2. Form the uncentered Hermitian Gram matrix `C = Hᴴ H`, matching classic PCA's
   covariance convention. Diagonalize it independently for this spatial block.
3. Compute `pᵢ = max(λᵢ, 0) / Σ max(λⱼ, 0)` and the entropy effective rank
   `e = exp(−Σ pᵢ ln pᵢ)` and choose `r = min(T, ceil(pca_rank_factor * e))`.
   Factor 1 preserves the original cutoff. Zero probabilities contribute zero entropy; a zero
   spectrum has rank zero. A tolerance of `1e-6` before rounding avoids promoting
   an integer effective rank due to floating-point roundoff.
4. Remove the strongest `r` components and reconstruct the residual
   `H V_keep V_keepᴴ` in the original time coordinates.
5. Add each residual sample to its original wrapped pixel/time coordinate, then
   divide by the actual number of spatial contributions at that pixel.

With 32×32 blocks and 16×16 strides, every pixel has four contributions when both
frame dimensions are divisible by 16. Other geometries can have varying overlap
counts, which are normalized individually. Blocks larger than the frame also wrap;
repeated samples contribute to the count.

Averaging occurs on complex residuals before the existing magnitude, squared
magnitude, or phase conversion. The existing P start/width controls therefore
select residual **time samples**, rather than independently ordered local PCA
components. Q bounds do not select components in this transform; scaled effective
rank does that automatically. With a positive factor, a one-frame block produces
a zero residual, as does any block whose selected removal count reaches `T`.

The implementation processes spatial blocks in batches, using batched
eigendecomposition and residual reconstruction. Covariance matrices retain the
original per-block matrix multiplication to preserve rounding behavior for weak
components. It reuses buffers for at most 64 blocks,
with a 64 MiB budget for tile scratch storage (excluding eigensolver workspace
and the coverage map); a single larger block can exceed that budget. Large blocks
that cannot be batched use the original single-block eigensolver.

Coverage is computed once for each geometry. When block sizes equal their strides
and divide the image dimensions exactly, residuals are written directly without
atomic accumulation or overlap normalization. Wrapped borders and oversized blocks
still use their actual contribution counts. CUDA and per-block eigensolver failures
are checked. Configuration and recording metadata store the five settings; older
configurations receive the defaults above, including factor 1.
