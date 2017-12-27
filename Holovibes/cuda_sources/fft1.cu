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
#include "preprocessing.cuh"
#include "transforms.cuh"
#include "unique_ptr.hh"

using camera::FrameDescriptor;

unsigned int binomial_coeff(unsigned int n, unsigned int k) {
	if (k == 0 || k == n)
		return 1;
	return binomial_coeff(n - 1, k - 1) + binomial_coeff(n - 1, k);
}

__global__
void kernel_compute_all_binomial_coeff(uint* coeffs, uint nb_coef)
{
	for (uint n = 0; n < nb_coef; n++) {
		coeffs[n * nb_coef] = 1;
		uint last_line = (n - 1) * nb_coef;
		for (uint k = 1; k <= n; k++) {
			coeffs[n * nb_coef + k] = coeffs[last_line + k - 1] + coeffs[last_line + k];
		}
	}
}

void fft1_lens(cuComplex*			lens,
	const FrameDescriptor&	fd,
	const float				lambda,
	const float				z,
	const float				pixel_size,
	cudaStream_t			stream)
{
	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(fd.frame_res(), threads);

	kernel_quadratic_lens << <blocks, threads, 0, stream >> >(lens, fd, lambda, z, pixel_size);
	cudaCheckError();
}

void fft1_lens_zernike(cuComplex*	lens,
		const FrameDescriptor&	fd,
		const float				lambda,
		const float				z,
		const float				pixel_size,
		const uint				zernike_m,
		const uint				zernike_n,
		cudaStream_t			stream)
{
	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(fd.frame_res(), threads);

	const auto nb_coef = zernike_n + 1;
	float coef = M_PI * lambda * z * 1E6;
	size_t size_coef = pow(nb_coef, 2);
	holovibes::cuda_tools::UniquePtr<unsigned int> binomial_coeffs(size_coef);
	kernel_compute_all_binomial_coeff << <1, 1, 0, stream >> > (binomial_coeffs, nb_coef);
	cudaCheckError();

	cudaStreamSynchronize(stream);

	kernel_zernike_polynomial << <blocks, threads, 0, stream >> > (lens, fd, pixel_size, coef, zernike_m, zernike_n, binomial_coeffs, nb_coef);
	cudaCheckError();
}

void fft_1(cuComplex*			input,
		const cuComplex*		lens,
		const cufftHandle		plan2D,
		const uint				frame_resolution,
		cudaStream_t			stream)
{
	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_resolution, threads);
	
	// Apply lens on multiple frames.
	kernel_apply_lens <<<blocks, threads, 0, stream>>>(input, frame_resolution, lens, frame_resolution);
	cudaCheckError();
	cudaStreamSynchronize(stream);
	// FFT
    cufftExecC2C(plan2D, input, input, CUFFT_FORWARD);

	cudaStreamSynchronize(stream);
}
