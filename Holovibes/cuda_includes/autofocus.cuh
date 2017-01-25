/*! \file 
 *
 * Function related to the autofocus computation. */
#pragma once

/*! \brief This function calculates the focus_metric value of a
 * given image, that will be then used in the pipe to find the best
 * one out of all ones and hence find the best z.
 */
float focus_metric(
  float* input,
  const unsigned int square_size,
  cudaStream_t stream,
  const unsigned int local_var_size);