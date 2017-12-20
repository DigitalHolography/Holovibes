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
	uint threads = 128;
	uint blocks = map_blocks_to_problem(fd.frame_res(), threads);

	const uint m = 1, n = 3;
	const auto nb_coef = n + 1;
	float coef = M_PI * lambda * z * 1E6;
	size_t size_coef = pow(nb_coef, 2);
	holovibes::cuda_tools::UniquePtr<unsigned int> binomial_coeffs(size_coef);
	kernel_compute_all_binomial_coeff << <1, 1, 0, 0 >> > (binomial_coeffs.get(), nb_coef);
	std::unique_ptr<uint[]> binom_cpu = std::make_unique<uint[]>(size_coef);
	cudaMemcpy(binom_cpu.get(), binomial_coeffs.get(), size_coef * sizeof(uint), cudaMemcpyDeviceToHost);

	//kernel_quadratic_lens << <blocks, threads, 0, stream >> >(lens, fd, lambda, z, pixel_size);
	//kernel_zernike_polynomial << <blocks, threads, 0, stream >> > (lens, fd, pixel_size, M_PI * lambda * z, 0, 4, binomial_coef.get(), nb_coef);

  cuComplex* output = lens;

  for (int index = 0; index < fd.width * fd.height; index++) {
	  const int i = index % fd.width;
	  const int j = index / fd.width;

	  const float	dx = pixel_size;// *1.0e-6f;
	  const float	dy = dx;

	  const float x = (i - fd.width / 2) / (fd.width / 2.f);
	  const float y = (j - fd.height / 2) / (fd.height / 2.f);

	  const float rho = hypotf(x, y); // Magnitude
	  const float phi = atan2(x, y);  // Argument

	  float Rmn = 0;
	  for (unsigned int k = 0; k <= (n - m) / 2; k++) {
		  float term = binom_cpu[(n - k) * nb_coef + k]
			  * binom_cpu[(n - 2 * k) * nb_coef + (n - m) / 2 - k]
			  * powf(rho, n - 2 * k);
		 /*float term = binomial_coeff(n - k, k)
			  * binomial_coeff(n - 2 * k, (n - m) / 2 - k)
			  * powf(rho, n - 2 * k);*/
		  if (k % 2)
			  Rmn -= term;
		  else
			  Rmn += term;
	  }
	  float Zmn = coef * Rmn * cos(m * phi);
	  cuComplex res = make_cuComplex(cosf(Zmn), sinf(Zmn));
	  cudaMemcpy(output + index, &res, sizeof(cuComplex), cudaMemcpyHostToDevice);
	  /*output[index].x = cosf(Zmn);
	  output[index].y = sinf(Zmn);*/
  }
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
