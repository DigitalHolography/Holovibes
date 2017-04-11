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

# include "autofocus.cuh"
# include "tools_compute.cuh"
# include "average.cuh"

static __global__
void kernel_minus_operator(const float	*input_left,
						const float		*input_right,
						float			*output,
						const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		output[index] = input_left[index] - input_right[index];
	}
}

static float global_variance_intensity(const float	*input, const uint	size)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	// <I>
	const float average_input = average_operator(input, size);

	// We create a matrix of <I> in order to do the substraction
	float	*matrix_average;
	size_t size_float = size * sizeof(float);
	if (cudaMalloc(&matrix_average, size_float) != cudaSuccess)
		return (0.f);
	float	*cpu_average_matrix = new float[size];
	for (uint i = 0; i < size; ++i)
		cpu_average_matrix[i] = average_input;

	cudaMemcpy(matrix_average, cpu_average_matrix, size_float, cudaMemcpyHostToDevice);

	delete[] cpu_average_matrix;
	// I - <I>
	kernel_minus_operator << <blocks, threads >> > (input, matrix_average, matrix_average, size);

	// We take it to the power of 2
	kernel_multiply_frames_float << <blocks, threads >> > (matrix_average, matrix_average, matrix_average, size);

	// And we take the average
	const float global_variance = average_operator(matrix_average, size);

	if (matrix_average)
		cudaFree(matrix_average);

	return global_variance;
}

static __global__
void kernel_float_to_complex(const float	*input,
							cuComplex		*output,
							const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		output[index].x = input[index];
		output[index].y = input[index];
	}
}

static float average_local_variance(const float	*input,
									const uint	square_size,
									const uint	mat_size)
{
	const uint	size = square_size * square_size;
	const uint	threads = get_max_threads_1d();
	const uint	blocks = map_blocks_to_problem(size, threads);

	/* ke matrix with same size than input */
	cuComplex	*ke_gpu_frame;
	size_t		ke_gpu_frame_pitch;

	/* Allocates memory for ke_gpu_frame. */
	const size_t square_size_complex = square_size * sizeof(cuComplex);
	if (cudaMallocPitch(&ke_gpu_frame,
		&ke_gpu_frame_pitch,
		square_size_complex,
		square_size) == cudaSuccess)
		cudaMemset2D(ke_gpu_frame,
			ke_gpu_frame_pitch,
			0,
			square_size_complex,
			square_size);
	else
		return (0.f);

	const uint square_mat_size = mat_size * mat_size;
	cuComplex* ke_complex_cpu = new cuComplex[square_mat_size];

	for (uint i = 0; i < square_mat_size; i++) // ++i before
	{
		ke_complex_cpu[i].x = 1 / static_cast<float>(square_mat_size);
		ke_complex_cpu[i].y = 1 / static_cast<float>(square_mat_size);
	}

	/* Copy the ke matrix to ke_gpu_frame. */
	const size_t mat_size_complex = mat_size * sizeof(cuComplex);
	cudaMemcpy2D(
		ke_gpu_frame,
		ke_gpu_frame_pitch,
		ke_complex_cpu,
		mat_size_complex,
		mat_size_complex,
		mat_size,
		cudaMemcpyHostToDevice);

	delete[] ke_complex_cpu;

	cuComplex *input_complex;
	if (cudaMalloc(&input_complex, size * sizeof(cuComplex)) != cudaSuccess)
	{
		cudaFree(ke_gpu_frame);
		return (0.f);
	}
	/* Convert input float frame to complex frame. */
	kernel_float_to_complex << <blocks, threads >> > (input, input_complex, size);

	/* Allocation of convolution i * ke output */
	float	*i_ke_convolution;
	if (cudaMalloc(&i_ke_convolution, size * sizeof(float)) != cudaSuccess)
	{
		cudaFree(input_complex);
		cudaFree(ke_gpu_frame);
		return (0.f);
	}

	cufftHandle plan2d_x;
	cufftHandle plan2d_k;
	cufftPlan2d(&plan2d_x, square_size, square_size, CUFFT_C2C);
	cufftPlan2d(&plan2d_k, square_size, square_size, CUFFT_C2C);

	/* Compute i * ke. */
	convolution_operator(input_complex,
		ke_gpu_frame,
		i_ke_convolution,
		size,
		plan2d_x,
		plan2d_k);

	/* Compute i - i * ke. */
	kernel_minus_operator << <blocks, threads >> > (input, i_ke_convolution, i_ke_convolution, size);

	/* Compute (i - i * ke)^2 */
	kernel_multiply_frames_float << <blocks, threads >> > (i_ke_convolution, i_ke_convolution, i_ke_convolution, size);

	cudaDeviceSynchronize();

	const float average_local_variance = average_operator(i_ke_convolution, size);

	/* -- Free ressources -- */
	cufftDestroy(plan2d_x);
	cufftDestroy(plan2d_k);

	cudaFree(i_ke_convolution);
	cudaFree(input_complex);
	cudaFree(ke_gpu_frame);

	return average_local_variance;
}

static __global__
void kernel_plus_operator(const float	*input_left,
						const float		*input_right,
						float			*output,
						const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		output[index] = input_left[index] + input_right[index];
	}
}

static __global__
void kernel_sqrt_operator(const float	*input,
						float			*output,
						const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		output[index] = sqrtf(input[index]);
	}
}

static float sobel_operator(const float	*input,
							const uint	square_size)
{
	const uint size = square_size * square_size;
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	/* ks matrix with same size than input */
	cuComplex	*ks_gpu_frame;
	size_t		ks_gpu_frame_pitch;

	const uint complex_square_size = square_size * sizeof(cuComplex);
	/* Allocates memory for ks_gpu_frame. */
	if (cudaMallocPitch(&ks_gpu_frame,
		&ks_gpu_frame_pitch,
		complex_square_size,
		square_size) == cudaSuccess)
		cudaMemset2D(ks_gpu_frame,
			ks_gpu_frame_pitch,
			0,
			complex_square_size,
			square_size);
	else
		return (0.f);

	/* Build the ks 3x3 matrix */
	const float ks_cpu[9] =
	{
		1.0f, 0.0f, -1.0f,
		2.0f, 0.0f, -2.0f,
		1.0f, 0.0f, -1.0f
	};

	cuComplex ks_complex_cpu[9];
	for (uint i = 0; i < 9; ++i)
	{
		ks_complex_cpu[i].x = ks_cpu[i];
		ks_complex_cpu[i].y = 0;//ks_cpu[i];
	}
	const uint complex_size3 = 3 * sizeof(cuComplex);
	/* Copy the ks matrix to ks_gpu_frame. */
	cudaMemcpy2D(ks_gpu_frame,
		ks_gpu_frame_pitch,
		ks_complex_cpu,
		complex_size3,
		complex_size3,
		3,
		cudaMemcpyHostToDevice);

	/* kst matrix with same size than input */
	cuComplex	*kst_gpu_frame;
	size_t		kst_gpu_frame_pitch;

	/* Allocates memory for kst_gpu_frame. */
	if (cudaMallocPitch(&kst_gpu_frame,
		&kst_gpu_frame_pitch,
		complex_square_size,
		square_size) == cudaSuccess)
		cudaMemset2D(kst_gpu_frame,
			kst_gpu_frame_pitch,
			0,
			complex_square_size,
			square_size);
	else
	{
		cudaFree(ks_gpu_frame);
		return (0.f);
	}
	/* Build the kst 3x3 matrix */
	const float kst_cpu[9] =
	{
		1.0f, 2.0f, 1.0f,
		0.0f, 0.0f, 0.0f,
		-1.0f, -2.0f, -1.0f
	};

	cuComplex kst_complex_cpu[9];
	for (uint i = 0; i < 9; ++i)
	{
		kst_complex_cpu[i].x = kst_cpu[i];
		kst_complex_cpu[i].y = kst_cpu[i];
	}

	/* Copy the kst matrix to kst_gpu_frame. */
	cudaMemcpy2D(kst_gpu_frame,
		kst_gpu_frame_pitch,
		kst_complex_cpu,
		complex_size3,
		complex_size3,
		3,
		cudaMemcpyHostToDevice);
	cuComplex	*input_complex;
	if (cudaMalloc(&input_complex, size * sizeof(cuComplex)) != cudaSuccess)
	{
		cudaFree(ks_gpu_frame);
		cudaFree(kst_gpu_frame);
		return (0.f);
	}

	/* Convert input float frame to complex frame. */
	kernel_float_to_complex << <blocks, threads >> > (input, input_complex, size);

	/* Allocation of convolution i * ks output */
	const int sizefloat = size * sizeof(float);
	float* i_ks_convolution;
	if (cudaMalloc(&i_ks_convolution, sizefloat) != cudaSuccess)
	{
		cudaFree(ks_gpu_frame);
		cudaFree(kst_gpu_frame);
		cudaFree(input_complex);
		return (0.f);
	}

	/* Allocation of convolution i * kst output */
	float *i_kst_convolution;
	if (cudaMalloc(&i_kst_convolution, sizefloat) != cudaSuccess)
	{
		cudaFree(ks_gpu_frame);
		cudaFree(kst_gpu_frame);
		cudaFree(input_complex);
		cudaFree(i_ks_convolution);
		return (0.f);
	}

	cufftHandle plan2d_x;
	cufftHandle plan2d_k;
	cufftPlan2d(&plan2d_x, square_size, square_size, CUFFT_C2C);
	cufftPlan2d(&plan2d_k, square_size, square_size, CUFFT_C2C);

	/* Compute i * ks. */
	convolution_operator(input_complex,
		ks_gpu_frame,
		i_ks_convolution,
		size,
		plan2d_x,
		plan2d_k);

	/* Compute (i * ks)^2 */
	kernel_multiply_frames_float << <blocks, threads >> > (i_ks_convolution, i_ks_convolution, i_ks_convolution, size);

	/* Compute i * kst. */
	convolution_operator(input_complex, kst_gpu_frame, i_kst_convolution, size, plan2d_x, plan2d_k);

	/* Compute (i * kst)^2 */
	kernel_multiply_frames_float << <blocks, threads >> > (i_kst_convolution, i_kst_convolution, i_kst_convolution, size);

	/* Compute (i * ks)^2 - (i * kst)^2 */
	kernel_plus_operator << <blocks, threads >> > (i_ks_convolution, i_kst_convolution, i_ks_convolution, size);

	kernel_sqrt_operator << <blocks, threads >> > (i_ks_convolution, i_ks_convolution, size);

	cudaDeviceSynchronize();

	const float average_magnitude = average_operator(i_ks_convolution, size);

	/* -- Free ressources -- */
	cufftDestroy(plan2d_x);
	cufftDestroy(plan2d_k);

	cudaFree(i_ks_convolution);
	cudaFree(i_kst_convolution);
	cudaFree(input_complex);
	cudaFree(kst_gpu_frame);
	cudaFree(ks_gpu_frame);

	return (1 / average_magnitude);
}

float focus_metric(	float			*input,
					const uint		square_size,
					cudaStream_t	stream,
					const uint		local_var_size)
{
	const uint	size = square_size * square_size;
	const uint	threads = get_max_threads_1d();
	const uint	blocks = map_blocks_to_problem(size, threads);

	/* Divide each pixels to avoid higher values than float can contains. */
	kernel_float_divide << <blocks, threads, 0, stream >> > (input, size, static_cast<float>(size));

	const float global_variance = global_variance_intensity(input, size);
	const float avr_local_variance = average_local_variance(input, square_size, local_var_size);
	const float avr_magnitude = sobel_operator(input, square_size);

	return (global_variance * avr_local_variance * avr_magnitude);
}
