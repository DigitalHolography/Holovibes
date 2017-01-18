/*! \file
 *
 * Functions that will compute the calculation of the average.
 */
#pragma once

# include "cuda_shared.cuh"
# include <tuple>

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
														  float							*input,
														  const uint					width,
														  const uint					height,
														  const holovibes::Rectangle&	signal,
														  const holovibes::Rectangle&	noise,
														  cudaStream_t					stream = 0);

/*! \brief  Make the average plot on the 2 select zones
* but first it will call the reconstruct roi after having
* splitted the image for the stft.
*/
std::tuple<float, float, float, float> make_average_stft_plot(complex*				cbuf,
															  float*                fbuf,
															  complex*				input,
															  const uint			width,
															  const uint			height,
															  const uint			width_roi,
															  const uint			height_roi,
															  holovibes::Rectangle& signal_zone,
															  holovibes::Rectangle& noise_zone,
															  const uint			pindex,
															  const uint			nsamples,
															  cudaStream_t			stream = 0);