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
 * \param input The input data to process in-place.
 * \param input_size Total number of elements to process. Should be a multiple
 * of lens_size.
 * \param lens The precomputed lens to apply.
 * \param lens_size The number of elements in the lens matrix.
 */
__global__
void kernel_apply_lens(cuComplex*		input,
					const uint			input_size,
					const cuComplex*	lens,
					const uint			lens_size);

/*! \brief Shifts in-place the corners of an image.
 *
 * This function shifts zero-frequency components to the center
 * of the spectrum (and vice-versa), as explained in the matlab documentation
 * (http://fr.mathworks.com/help/matlab/ref/fftshift.html).
 *
 * \param input The image to modify in-place.
 * \param size_x The width of data, in pixels.
 * \param size_y The height of data, in pixels.
 * \param stream The CUDA stream on which to launch the operation.
 */
void shift_corners(float*		input,
				const uint		size_x,
				const uint		size_y,
				cudaStream_t	stream = 0);

/*! \brief Compute the log base-10 of every element of the input.
*
* \param input The image to modify in-place.
* \param size The number of elements to process.
* \param stream The CUDA stream on which to launch the operation.
*/
void apply_log10(float*			input,
				const uint		size,
				cudaStream_t	stream = 0);

/*! \brief Allows demodulation in real time. Considering that we need (exactly like
*   an STFT) put every pixel in a particular order to apply an FFT and then reconstruct
*   the frame, please consider that this computation is very costly.

* !!! An explanation of how the computation is given in stft.cuh !!!

* \param input input buffer is where frames are taken for computation
* \param stft_buf the buffer which will be exploded
* \param stft_dup_buf the buffer that will receive the plan1d transforms
* \parem frame_resolution number of pixels in one frame.
* \param nSize number of frames that will be used.

*/
void demodulation(cuComplex*		input,
				const cufftHandle	plan,
				cudaStream_t		stream = 0);

/*! \brief Apply the convolution operator to 2 complex matrices.
*
* \param First input matrix
* \param Second input matrix
* \param out Output matrix storing the result of the operation
* \param The number of elements in each matrix
* \param plan2d_x Externally prepared plan for x
* \param plan2d_k Externally prepared plan for k
* \param stream The CUDA stream on which to launch the operation.
*/
void convolution_operator(const cuComplex*	x,
						const cuComplex*	k,
						float*				out,
						const uint			size,
						const cufftHandle	plan2d_x,
						const cufftHandle	plan2d_k,
						cudaStream_t		stream = 0);

void convolution_float(const float			*a,
						const float			*b,
						float				*out,
						const uint			size,
						const cufftHandle	plan2d_a,
						const cufftHandle	plan2d_b,
						const cufftHandle	plan2d_invert,
						cudaStream_t		stream = 0);

/*! \brief Extract a part of the input image to the output.
*
* \param input The full input image
* \param zone the part of the image we want to extract
* \param In pixels, the original width of the image
* \param Where to store the cropped image
* \param output_width In pixels, the desired width of the cropped image
* \param stream The CUDA stream on which to launch the operation.
*/
void frame_memcpy(float*			input,
				const holovibes::units::RectFd&	zone,
				const uint			input_width,
				float*				output,
				const uint			output_width,
				cudaStream_t		stream = 0);

/*! \brief Make the average of every element contained in the input.
 *
 * \param input The input data to average.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 *
 * \return The average value of the *size* first elements.
 */
float average_operator(const float*	input,
					const uint		size,
					cudaStream_t	stream = 0);

/*! \brief Make the average of every element contained in the input.
 *
 * \param input The input data to average in cufftComplex.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 *
 * \return The average value of the *size* first elements.
 */
float average_operator_from_complex(const cufftComplex *input,
					const uint		size,
					cudaStream_t	stream = 0);

/*! Let H be the latest complex image, H-t the conjugate matrix of
* the one preceding it, and .* the element-to-element matrix
* multiplication operation.
* This version computes : arg(H(t) .* H^*(t- T))
*
* Phase increase adjusts phase angles encoded in complex data,
* by a cutoff value (which is here fixed to pi). Unwrapping seeks
* two-by-two differences that exceed this cutoff value and performs
* cumulative adjustments in order to 'smooth' the signal.
*/
void phase_increase(const cuComplex*		cur,
					holovibes::UnwrappingResources*	resources,
					const size_t			image_size);

/*! Main function for unwrap_2d calculations*/
void unwrap_2d(float*					input,
			const cufftHandle			plan2d,
			holovibes::UnwrappingResources_2d*	res,
			const camera::FrameDescriptor&		fd,
			float*						output,
			cudaStream_t				stream = 0);

/*! Gradient calculation for unwrap_2d calculations*/
void gradient_unwrap_2d(const cufftHandle		plan2d,
						holovibes::UnwrappingResources_2d*	res,
						const camera::FrameDescriptor&		fd,
						cudaStream_t			stream);

/*! Eq calculation for unwrap_2d calculations*/
void eq_unwrap_2d(const cufftHandle		plan2d,
				holovibes::UnwrappingResources_2d*	res,
				const camera::FrameDescriptor&		fd,
				cudaStream_t			stream);

/*! Phi calculation for unwrap_2d calculations*/
void phi_unwrap_2d(const cufftHandle	plan2d,
				holovibes::UnwrappingResources_2d*	res,
				const camera::FrameDescriptor&		fd,
				float*					output,
				cudaStream_t			stream);

/*  \brief Circularly shifts the elements in input given a point(i,j)
**   and the size of the frame.
*/
__global__
void circ_shift(cuComplex*	input,
				cuComplex*	output,
				const int	i, // shift on x axis
				const int	j, // shift on y axis
				const uint	width,
				const uint	height,
				const uint	size);

/*  \brief Circularly shifts the elements in input given a point(i,j)
 *	given float output & inputs.
 */
__global__
void circ_shift_float(float*	input,
					float*		output,
					const int	i, // shift on x axis
					const int	j, // shift on y axis
					const uint	width,
					const uint	height,
					const uint	size);

/* \brief Translates the image by shift_x and shift_y
 *
 */
void complex_translation(float		*frame,
						uint		width,
						uint		height,
						int			shift_x,
						int			shift_y);

void correlation_operator(float* a, float* b, float* out, QPoint dimensions);

__global__
void kernel_complex_to_modulus(const cuComplex	*input,
							float				*output,
							const uint			size);

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void normalize_float_matrix(const float *g_idata, float *g_odata, unsigned int n) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n) { sdata[tid] += abs(g_idata[i]) + abs(g_idata[i + blockSize]); i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* \brief return the sum of the abs of values in the matrix
 *
 */
float get_norm(const float	*matrix,
			   size_t		size);