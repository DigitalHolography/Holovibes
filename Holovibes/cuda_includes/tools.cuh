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

# include "tools.hh"
# include "compute_bundles.hh"
# include "compute_bundles_2d.hh"


/*! \brief  Apply a previously computed lens to image(s).
 *
 * The input data is multiplied element-wise with each corresponding
 * lens coefficient.
 *
 * \param input The input data to process.
 * \param output The output data processed
 * \param batch_size The number of frames in input
 * \param input_size Total number of elements to process. Should be a multiple
 * of lens_size.
 * \param lens The precomputed lens to apply.
 * \param lens_size The number of elements in the lens matrix.
 */
__global__
void kernel_apply_lens(cuComplex		*input,
					cuComplex 			*output,
					const uint 			batch_size,
					const uint			input_size,
					const cuComplex*	lens,
					const uint			lens_size);

/*! \brief Extract a part of the input image to the output.
*
* \param input The full input image
* \param zone the part of the image we want to extract
* \param In pixels, the original width of the image
* \param Where to store the cropped image
* \param stream The CUDA stream on which to launch the operation.
*/
void frame_memcpy(const float*			input,
				const holovibes::units::RectFd&	zone,
				const uint			input_width,
				float*				output,
				const cudaStream_t		stream = 0);


/*  \brief Circularly shifts the elements in input given a point(i,j)
**   and the size of the frame.
*/
__global__
void circ_shift(const cuComplex*	input,
				cuComplex*	output,
				const uint 	batch_size,
				const int	i, // shift on x axis
				const int	j, // shift on y axis
				const uint	width,
				const uint	height,
				const uint	size);

/*  \brief Circularly shifts the elements in input given a point(i,j)
 *	given float output & inputs.
 */
__global__
void circ_shift_float(const float*	input,
					float*		output,
					const uint 	batch_size,
					const int	i, // shift on x axis
					const int	j, // shift on y axis
					const uint	width,
					const uint	height,
					const uint	size);

__global__
void kernel_complex_to_modulus(const cuComplex	*input,
							float				*output,
							const uint			size);