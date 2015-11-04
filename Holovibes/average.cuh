/*! \file */
#ifndef AVERAGE_CUH
# define AVERAGE_CUH

# include <tuple>
# include "geometry.hh"
# include <cufft.h>

/*! \brief  Make the average plot on the 2 select zones
*
* \param input The image from where zones should be ploted
* \param width The width of the input image.
* \param height The height of the input image.
* \param signal Coordinates of the signal zone to use.
* \param noise Coordinates of the noise zone to use.
* \rerun A tupple of 3 floats <sum of signal zones pixels, sum of noise zone pixels, average>.
*
*/
std::tuple<float, float, float> make_average_plot(
  float *input,
  const unsigned int width,
  const unsigned int height,
  holovibes::Rectangle& signal,
  holovibes::Rectangle& noise);

/*! \brief  Make the average plot on the 2 select zones
* but first it will call the reconstruct roi after having
* splitted the image for the stft.
*/
std::tuple<float, float, float> make_average_stft_plot(
  cufftComplex*          cbuf,
  float*                 fbuf,
  cufftComplex*          input,
  unsigned int           width,
  unsigned int           height,
  unsigned int           width_roi,
  unsigned int           height_roi,
  holovibes::Rectangle&  signal_zone,
  holovibes::Rectangle&  noise_zone,
  unsigned int           pindex,
  unsigned int           nsamples);

#endif /* !AVERAGE_CUH */