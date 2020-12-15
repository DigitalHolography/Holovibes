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

#include <exception>

#include <cuda_runtime.h>

#include "compute_bundles_2d.hh"
#include "cuda_memory.cuh"

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
		cudaXMalloc(&gpu_fx_, sizeof(float) * image_resolution_);
		cudaXMalloc(&gpu_fy_, sizeof(float) * image_resolution_);
		cudaXMalloc(&gpu_shift_fx_, sizeof(float) * image_resolution_);
		cudaXMalloc(&gpu_shift_fy_, sizeof(float) * image_resolution_);
		cudaXMalloc(&gpu_angle_, sizeof(float) * image_resolution_);
		cudaXMalloc(&gpu_z_, sizeof(cufftComplex) * image_resolution_);
		cudaXMalloc(&gpu_grad_eq_x_, sizeof(cufftComplex) * image_resolution_);
		cudaXMalloc(&gpu_grad_eq_y_, sizeof(cufftComplex) * image_resolution_);
		minmax_buffer_ = new float[image_resolution_]();
	}

	UnwrappingResources_2d::~UnwrappingResources_2d()
	{
		cudaXFree(gpu_fx_);
		cudaXFree(gpu_fy_);
		cudaXFree(gpu_shift_fx_);
		cudaXFree(gpu_shift_fy_);
		cudaXFree(gpu_angle_);
		cudaXFree(gpu_z_);
		cudaXFree(gpu_grad_eq_x_);
		cudaXFree(gpu_grad_eq_y_);
		delete[] minmax_buffer_;
	}

	void UnwrappingResources_2d::cudaRealloc(void *ptr, const size_t size)
	{
		cudaXFree(ptr);
		cudaXMalloc(&ptr, size);
	}

	void UnwrappingResources_2d::reallocate(const size_t image_size)
	{
		image_resolution_ = image_size;

		cudaRealloc(gpu_fx_, sizeof(float) * image_resolution_);
		cudaRealloc(gpu_fy_, sizeof(float) * image_resolution_);
		cudaRealloc(gpu_shift_fx_, sizeof(float) * image_resolution_);
		cudaRealloc(gpu_shift_fy_, sizeof(float) * image_resolution_);
		cudaRealloc(gpu_angle_, sizeof(float) * image_resolution_);
		cudaRealloc(gpu_z_, sizeof(cufftComplex) * image_resolution_);
		cudaRealloc(gpu_grad_eq_x_, sizeof(cufftComplex) * image_resolution_);
		cudaRealloc(gpu_grad_eq_y_, sizeof(cufftComplex) * image_resolution_);
		if (minmax_buffer_)
			delete[] minmax_buffer_;
		minmax_buffer_ = nullptr;
		minmax_buffer_ = new float[image_resolution_]();
	}
}