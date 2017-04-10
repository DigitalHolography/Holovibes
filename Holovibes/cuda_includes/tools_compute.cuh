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

/*! \brief  Divide all the pixels of input image(s) by the float divider.
*
* \param image The image(s) to process. Should be contiguous memory.
* \param size Number of elements to process.
* \param divider Divider value for all elements.
*/
__global__
void kernel_complex_divide(cuComplex	*image,
						const uint		size,
						const float		divider);

/*! \brief  Divide all the pixels of input image(s) by the float divider.
*
* \param image The image(s) to process. Should be contiguous memory.
* \param size Number of elements to process.
* \param divider Divider value for all elements.
*/
__global__
void kernel_float_divide(float		*input,
						const uint	size,
						const float	divider);

/*! \brief  Multiply the pixels value of 2 complexe input images
*
* The images to multiply should have the same size.
* The result is given in output.
* Output should have the same size of inputs.
*/
__global__
void kernel_multiply_frames_complex(const cuComplex	*input1,
									const cuComplex	*input2,
									cuComplex		*output,
									const uint		size);

/*! \brief  Multiply the pixels value of 2 float input images
*
* The images to multiply should have the same size.
* The result is given in output.
* Output should have the same size of inputs.
*/
__global__
void kernel_multiply_frames_float(const float	*input1,
								const float		*input2,
								float			*output,
								const uint		size);

/*! \brief kernel wich compute the substract of a reference image to another */
__global__
void kernel_substract_ref(cuComplex				*input,
						void					*reference,
						const ComputeDescriptor	cd,
						const uint				nframes);

/*! \brief  substract the pixels value of a reference image to another
*
* The images to multiply should have the same size.
* The result is given in output.
* Output should have the same size of inputs.
*/
void substract_ref(cuComplex	*input,
				cuComplex		*reference,
				const uint		frame_resolution,
				const uint		nframes,
				cudaStream_t	stream = 0);

/* \brief  Compute the mean of several images from output. The result image is put into output*/
void mean_images(cuComplex		*input,
				cuComplex		*output,
				uint			n,
				uint			frame_size,
				cudaStream_t	stream = 0);
