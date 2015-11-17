/*! \file */
#pragma once
# include <cufft.h>

/*! \brief For each pixel (P and Q) of the two images, this function
* will output on output (O) : \n
* Ox = (Px Qx + Py Qy) / (QxQx + QyQy) \n
* Oy = (Py Qx - Px Qy) / (QxQx + QyQy) \n
*
* \param frame_p the numerator image
* \param frame_q the denominator image
*/
void frame_ratio(
  const cufftComplex* frame_p,
  const cufftComplex* frame_q,
  cufftComplex* output,
  const unsigned int size);