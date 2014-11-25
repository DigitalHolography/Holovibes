#ifndef CONTRAST_CORRECTION_CUH
# define CONTRAST_CORRECTION_CUH

void manual_contrast_correction(
  float* input,
  unsigned int size,
  unsigned short dynamic_range,
  unsigned short min,
  unsigned short max);
void auto_contrast_correction(
  float* input,
  unsigned int size,
  unsigned int* min,
  unsigned int* max,
  float threshold);

#endif /* !CONTRAST_CORRECTION_CUH */