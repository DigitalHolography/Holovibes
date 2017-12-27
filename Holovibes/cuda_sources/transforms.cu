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

#include "transforms.cuh"

using camera::FrameDescriptor;

__global__
void kernel_quadratic_lens(cuComplex*			output,
						const FrameDescriptor	fd,
						const float				lambda,
						const float				dist,
						const float				pixel_size)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint	size = fd.width * fd.height;
	const float	c = M_PI / (lambda * dist);
	const float	dx = pixel_size * 1.0e-6f;
	const float	dy = dx;
	float		x, y;
	uint		i, j;
	float		csquare;

	if (index < size)
	{
		i = index % fd.width;
		j = index / fd.height;
		x = (i - static_cast<float>(fd.width >> 1)) * dx;
		y = (j - static_cast<float>(fd.height >> 1)) * dy;

		csquare = c * (x * x + y * y);
		output[index].x = cosf(csquare);
		output[index].y = sinf(csquare);
	}
}

__global__
void kernel_zernike_polynomial(cuComplex * output,
	const FrameDescriptor fd,
	const float pixel_size,
	const float coef,
	const unsigned int m, const unsigned int n,
	const uint* binomial_coeff, uint nb_coef)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint	size = fd.width * fd.height;
	
	if (index < size) {
		const int i = index % fd.width;
		const int j = index / fd.width;

		const float x = (i - fd.width / 2) / (fd.width / 2.f);
		const float y = (j - fd.height / 2) / (fd.height / 2.f);

		const float rho = hypotf(x, y); // Magnitude
		const float phi = atan2(x, y);  // Argument

		float Rmn = 0;
		for (unsigned int k = 0; k <= (n - m) / 2; k++) {
			float term = binomial_coeff[(n - k) * nb_coef + k]
				* binomial_coeff[(n - 2 * k) * nb_coef + (n - m) / 2 - k]
				* powf(rho, n - 2 * k);
			/*float term = binomial_coeff(n - k, k)
			* binomial_coeff(n - 2 * k, (n - m) / 2 - k)
			* powf(rho, n - 2 * k);*/
			Rmn += k % 2 ? -term : term;
		}
		float Zmn = coef * Rmn * cos(m * phi);
		cuComplex res = make_cuComplex(cosf(Zmn), sinf(Zmn));
		output[index].x = cosf(Zmn);
		output[index].y = sinf(Zmn);
	}
}

/*__device__
unsigned int binomial_coeff(unsigned int n, unsigned int k) {
	if (k == 0 || k == n)
		return 1;
	return binomial_coeff(n - 1, k - 1) + binomial_coeff(n - 1, k);
}*/

__global__
void kernel_spectral_lens(cuComplex				*output,
						const FrameDescriptor	fd,
						const float				lambda,
						const float				distance,
						const float				pixel_size)
{
	const uint	i = blockIdx.x * blockDim.x + threadIdx.x;
	const uint	j = blockIdx.y * blockDim.y + threadIdx.y;
	const uint	index = j * blockDim.x * gridDim.x + i;
	const float c = M_2PI * distance / lambda;
	const float dx = pixel_size * 1.0e-6f;
	const float dy = dx;
	const float du = 1 / ((static_cast<float>(fd.width)) * dx);
	const float dv = 1 / ((static_cast<float>(fd.height)) * dy);
	const float u = (i - static_cast<float>(lrintf(static_cast<float>(fd.width >> 1)))) * du;
	const float	v = (j - static_cast<float>(lrintf(static_cast<float>(fd.height >> 1)))) * dv;

	if (index < fd.width * fd.height)
	{
		const float lambda2 = lambda * lambda;
		const float csquare = c * sqrtf(abs(1.0f - lambda2 * u * u - lambda2 * v * v));
		output[index].x = cosf(csquare);
		output[index].y = sinf(csquare);
	}
}
