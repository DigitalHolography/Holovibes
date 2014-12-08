#ifndef CONTRAST_CORRECTION_CUH
# define CONTRAST_CORRECTION_CUH

void manual_contrast_correction(
  float* input,
  unsigned int size,
  unsigned short dynamic_range,
  float min,
  float max);
void auto_contrast_correction(
  float* input,
  unsigned int size,
  float* min,
  float* max);

#endif /* !CONTRAST_CORRECTION_CUH */