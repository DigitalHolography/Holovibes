#ifndef CONTRAST_CORRECTION_CUH
# define CONTRAST_CORRECTION_CUH

void manual_contrast_correction(
  void *img,
  unsigned int img_size,
  int bytedepth,
  unsigned int manual_min,
  unsigned int manual_max);
void auto_contrast_correction(
  unsigned int *min,
  unsigned int *max,
  void *img,
  unsigned int img_size,
  unsigned int bytedepth,
  unsigned int percent);

#endif /* !CONTRAST_CORRECTION_CUH */