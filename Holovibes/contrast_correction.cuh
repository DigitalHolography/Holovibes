/*! \file */
#pragma once

/*! \brief Make the contrast of the image depending of the
* maximum and minimum input given by the user.
*
* The algortihm used is a contrast stretching, the
* values min and max can be found thanks to the previous functions
* or can be set by the user in case of a particular use.
* This image should be stored in device VideoRAM.
* \param input The image to correct correct contrast.
* \param size Size of the image in number of pixels.
* \param min Minimum pixel value of the input image.
* \param max Maximum pixel value of the input image.
*
*/
void manual_contrast_correction(
  float* input,
  const unsigned int size,
  const unsigned short dynamic_range,
  const float min,
  const float max);

/*! \brief Find the minimum pixel value of an image and the maximum one.
*
* \param input The image to correct correct contrast.
* \param size Size of the image in number of pixels.
* \param min Minimum pixel value found.
* \param max Maximum pixel value found.
*
*/
void auto_contrast_correction(
  float* input,
  const unsigned int size,
  float* min,
  float* max);