/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once

# include "Common.cuh"

/*! \brief  Make the chart plot on the 2 selected zones
*
* \param width The width of the input image.
* \param height The height of the input image.
* \param signal Coordinates of the signal zone to use.
* \param noise Coordinates of the noise zone to use.
* \return A tuple of 4 floats <sum of signal zones pixels,
*			sum of noise zone pixels, average, log>.
*/
holovibes::Tuple4f make_chart_plot(float*			input,
						const uint			width,
						const uint			height,
						const holovibes::units::RectFd&	signal,
						const holovibes::units::RectFd&	noise,
						cudaStream_t		stream = 0);