/*! \file
 *
 * Functions that will compute the calculation of the average.
 */
#pragma once

# include <tuple>
# include <cuda_runtime.h>
# include <cufft.h>

/* Forward declaration. */
namespace holovibes
{
  struct Rectangle;
}

/*! \brief  Make the average plot on the 2 selected zones
*
* \param width The width of the input image.
* \param height The height of the input image.
* \param signal Coordinates of the signal zone to use.
* \param noise Coordinates of the noise zone to use.
* \return A tupple of 3 floats <sum of signal zones pixels, sum of noise zone pixels, average>.
*
*/
std::tuple<float, float, float, float> make_average_plot(
  float* input,
  const unsigned int width,
  const unsigned int height,
  const holovibes::Rectangle& signal,
  const holovibes::Rectangle& noise,
  cudaStream_t stream = 0);

/*! \brief  Make the average plot on the 2 select zones
* but first it will call the reconstruct roi after having
* splitted the image for the stft.
*/
std::tuple<float, float, float, float> make_average_stft_plot(
  cufftComplex*          cbuf,
  float*                 fbuf,
  cufftComplex*          input,
  const unsigned int     width,
  const unsigned int     height,
  const unsigned int     width_roi,
  const unsigned int     height_roi,
  holovibes::Rectangle&  signal_zone,
  holovibes::Rectangle&  noise_zone,
  const unsigned int     pindex,
  const unsigned int     nsamples,
  cudaStream_t stream = 0);