# Settings
## List of all holovibes API settings
### Advanced settings
- display_rate
- input_buffer_size
- record_buffer_size
- output_buffer_size
- raw_bitshift
- contrast_lower_threshold
- contrast_upper_threshold
- cuts_contrast_p_offset

### Compute settings
- batch_size
- time_stride
- time_transformation_size
- space_transformation
- time_transformation
- lambda
- z_distance
- convolution_enabled
- convolution_matrix
- convolution_divide
- input_fps
- compute_mode
- pixel_size
- unwrap_history_size
- is_computation_stopped
- time_transformation_cuts_output_buffer_size

#### Classic PCA effective rank

Classic PCA uses its original component scores by default. To enable automatic
component removal, import a compute-settings JSON file with the **Import compute
settings** menu or pass it to the CLI with `--compute_settings`.

```json
{
  "version": "v1",
  "image_rendering": {
    "time_transformation": "PCA",
    "pca_effective_rank": { "enabled": true, "factor": 1.0 }
  }
}
```

Classic PCA multiplies the full-frame effective rank by the factor and rounds
up to select how many of the strongest components to zero, up to the time
window size. The P controls select the same component indices. The factor
defaults to 1 and accepts finite, nonnegative values. A factor of 0 skips
component removal.

### Composite settings
- composite_kind
- composite_auto_weights
- rgb
- hsv

### Export settings
- frame_record_enabled
- chart_record_enabled

### Import settings
- start_frame
- end_frame

### View settings
- x
- y
- p
- q
- xy
- xz
- yz
- filter2d
- current_window
- lens_view_enabled
- chart_display_enabled
- filter2d_enabled
- fiter2d_view_enabled
- fft_shift_enabled
- raw_view_enabled
- cuts_view_enabled
- renorm_enabled
- reticle_scale
- reticle_display_enabled

### Filter2d settings
- filter2d_n1
- filter2d_n2
- filter2d_smooth_low
- filter2d_smooth_high

### FileRead settins
- file_buffer_size

### Zone settings
- signal_zone
- noise_zone
- composite_zone
- zoomed_zone
- reticle_zone
