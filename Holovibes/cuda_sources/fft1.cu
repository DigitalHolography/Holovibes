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

#include "fft1.cuh"
#include "transforms.cuh"
#include "unique_ptr.hh"
#include "Common.cuh"
#include "cuda_memory.cuh"

#include <cufftXt.h>

using camera::FrameDescriptor;


void fft1_lens(cuComplex*	lens,
	const uint 				lens_side_size,
	const uint 				frame_height,
	const uint 				frame_width,
	const float				lambda,
	const float				z,
	const float				pixel_size,
	cudaStream_t			stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(lens_side_size * lens_side_size, threads);

	cuComplex* square_lens;
	// In anamorphic mode, the lens is initally a square, it's then cropped to be the same dimension as the frame
	if (frame_height != frame_width)
		cudaXMalloc((void**)&square_lens, lens_side_size * lens_side_size * sizeof(cuComplex));
	else
		square_lens = lens;

	kernel_quadratic_lens<<<blocks, threads, 0, stream>>>(square_lens, lens_side_size, lambda, z, pixel_size);
	cudaCheckError();

	if (frame_height != frame_width)
	{
		// Data is contiguous for a horizontal frame so a simple memcpy with an offset and a limited size works
		if (frame_width > frame_height)
			cudaXMemcpy(lens, square_lens + ((lens_side_size - frame_height) / 2) * frame_width, frame_width * frame_height * sizeof(cuComplex));
		else
		{
			// For a vertical frame we need memcpy 2d to copy row by row, taking the offset into account every time
			cudaSafeCall(cudaMemcpy2D(lens,													// Destination (frame)
									  frame_width * sizeof(cuComplex),						// Destination width in byte
									  square_lens + ((lens_side_size - frame_width) / 2),	// Source (lens)
									  lens_side_size * sizeof(cuComplex),					// Source width in byte
									  frame_width * sizeof(cuComplex),						// Destination width in byte (yes it's redoundant)
									  frame_height,											// Destination height (not in byte)
									  cudaMemcpyDeviceToDevice));
		}
		cudaXFree(square_lens);
	}
}

void fft_1(cuComplex*			input,
		cuComplex* 				output,
		const uint 				batch_size,
		const cuComplex*		lens,
		const cufftHandle		plan2D,
		const uint				frame_resolution,
		cudaStream_t			stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_resolution, threads);

	// Apply lens on multiple frames.
	kernel_apply_lens <<<blocks, threads, 0, stream>>>(input, output, batch_size, frame_resolution, lens, frame_resolution);

	// No sync needed between kernel call and cufft call
	cudaCheckError();
	// FFT

	cufftSafeCall(cufftXtExec(plan2D, input, output, CUFFT_FORWARD));
	// Same, no sync needed since everything is executed on the stream 0

	cudaCheckError();
}

