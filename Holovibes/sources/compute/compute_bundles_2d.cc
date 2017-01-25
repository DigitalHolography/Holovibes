#include "compute_bundles_2d.hh"

namespace holovibes
{
	UnwrappingResources_2d::UnwrappingResources_2d(
		const size_t image_size)
		: image_resolution_(image_size)
		, gpu_fx_(nullptr)
		, gpu_fy_(nullptr)
		, gpu_z_(nullptr)
		, gpu_grad_eq_x_(nullptr)
		, gpu_grad_eq_y_(nullptr)
		, gpu_angle_(nullptr)
		, gpu_shift_fx_(nullptr)
		, gpu_shift_fy_(nullptr)
		, minmax_buffer_(nullptr)
	{
		cudaMalloc(&gpu_fx_, sizeof(float)* image_resolution_);

		cudaMalloc(&gpu_fy_, sizeof(float)* image_resolution_);

		cudaMalloc(&gpu_shift_fx_, sizeof(float)* image_resolution_);

		cudaMalloc(&gpu_shift_fy_, sizeof(float)* image_resolution_);

		cudaMalloc(&gpu_angle_, sizeof(float)* image_resolution_);

		cudaMalloc(&gpu_z_, sizeof(cufftComplex)* image_resolution_);

		cudaMalloc(&gpu_grad_eq_x_, sizeof(cufftComplex)* image_resolution_);

		cudaMalloc(&gpu_grad_eq_y_, sizeof(cufftComplex)* image_resolution_);

		minmax_buffer_ = new float[image_resolution_]();
	}

	UnwrappingResources_2d::~UnwrappingResources_2d()
	{
		if (gpu_fx_)
			cudaFree(gpu_fx_);
		if (gpu_fy_)
			cudaFree(gpu_fy_);
		if (gpu_shift_fx_)
			cudaFree(gpu_shift_fx_);
		if (gpu_shift_fy_)
			cudaFree(gpu_shift_fy_);
		if (gpu_angle_)
			cudaFree(gpu_angle_);
		if (gpu_z_)
			cudaFree(gpu_z_);
		if (gpu_grad_eq_x_)
			cudaFree(gpu_grad_eq_x_);
		if (gpu_grad_eq_y_)
			cudaFree(gpu_grad_eq_y_);
		if (minmax_buffer_)
			delete[] minmax_buffer_;
	}

	void UnwrappingResources_2d::reallocate(const size_t image_size)
	{
		image_resolution_ = image_size;

		if (gpu_fx_)
			cudaFree(gpu_fx_);
		cudaMalloc(&gpu_fx_, sizeof(float)* image_resolution_);

		if (gpu_fy_)
			cudaFree(gpu_fy_);
		cudaMalloc(&gpu_fy_, sizeof(float)* image_resolution_);

		if (gpu_shift_fx_)
			cudaFree(gpu_shift_fx_);
		cudaMalloc(&gpu_shift_fx_, sizeof(float)* image_resolution_);

		if (gpu_shift_fy_)
			cudaFree(gpu_shift_fy_);
		cudaMalloc(&gpu_shift_fy_, sizeof(float)* image_resolution_);

		if (gpu_angle_)
			cudaFree(gpu_angle_);
		cudaMalloc(&gpu_angle_, sizeof(float)* image_resolution_);

		if (gpu_z_)
			cudaFree(gpu_z_);
		cudaMalloc(&gpu_z_, sizeof(cufftComplex)* image_resolution_);

		if (gpu_grad_eq_x_)
			cudaFree(gpu_grad_eq_x_);
		cudaMalloc(&gpu_grad_eq_x_, sizeof(cufftComplex)* image_resolution_);

		if (gpu_grad_eq_y_)
			cudaFree(gpu_grad_eq_y_);
		cudaMalloc(&gpu_grad_eq_y_, sizeof(cufftComplex)* image_resolution_);

		if (minmax_buffer_)
			delete[] minmax_buffer_;
		minmax_buffer_ = new float[image_resolution_]();
	}
}