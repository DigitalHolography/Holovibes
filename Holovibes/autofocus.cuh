/*! \file */
#ifndef AUTOFOCUS_CUH
# define AUTOFOCUS_CUH

/*! \brief This function calculates the focus_metric value of a
 * given image, that will be then used in the pipeline to find the best
 * one out of all ones and hence find the best z.
 */
float focus_metric(
  float* input,
  unsigned int square_size);

#endif /* !AUTOFOCUS_CUH */