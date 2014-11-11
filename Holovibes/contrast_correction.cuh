#ifndef CONTRAST_CORRECTION_CUH
# define CONTRAST_CORRECTION_CUH

void correct_contrast(unsigned short *img, int img_size, int bytedepth);
void sum_histo_c(int *histo, int *summed_histo);

#endif /* !CONTRAST_CORRECTION_CUH */