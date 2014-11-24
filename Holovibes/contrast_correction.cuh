#ifndef CONTRAST_CORRECTION_CUH
# define CONTRAST_CORRECTION_CUH

void manual_contrast_correction(
  float* input,
  unsigned int size,
  unsigned int dynamic_range,
  unsigned int min,
  unsigned int max);
void auto_contrast_correction(
  unsigned int *min,
  unsigned int *max,
  void *img,
  unsigned int img_size,
  unsigned int bytedepth,
  unsigned int percent);

#endif /* !CONTRAST_CORRECTION_CUH */