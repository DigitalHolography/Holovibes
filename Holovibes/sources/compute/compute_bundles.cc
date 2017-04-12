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

#include "compute_bundles.hh"
#include <cufft.h>
#include <iostream>

namespace holovibes
{
	UnwrappingResources::UnwrappingResources(
		const unsigned capacity,
		const size_t image_size)
		: total_memory_(capacity)
		, capacity_(capacity)
		, size_(0)
		, next_index_(0)
		, gpu_unwrap_buffer_(nullptr)
		, gpu_predecessor_(nullptr)
		, gpu_angle_predecessor_(nullptr)
		, gpu_angle_current_(nullptr)
		, gpu_angle_copy_(nullptr)
		, gpu_unwrapped_angle_(nullptr)
	{
		int		err = 0;
		auto	nb_unwrap_elts = image_size * capacity_;

		if (cudaMalloc(&gpu_unwrap_buffer_, sizeof(float) * nb_unwrap_elts) != cudaSuccess)
			err++;
		if (cudaMalloc(&gpu_predecessor_, sizeof(cufftComplex) * image_size) != cudaSuccess)
			err++;
		if (cudaMalloc(&gpu_angle_predecessor_, sizeof(float) * image_size) != cudaSuccess)
			err++;
		if (cudaMalloc(&gpu_angle_current_, sizeof(float) * image_size) != cudaSuccess)
			err++;
		if (cudaMalloc(&gpu_angle_copy_, sizeof(float) * image_size) != cudaSuccess)
			err++;
		if (cudaMalloc(&gpu_unwrapped_angle_, sizeof(float) * image_size) != cudaSuccess)
			err++;
		if (err != 0)
			throw std::exception("Cannot allocate UnwrappingResources");
		/* Cumulative phase adjustments in gpu_unwrap_buffer are reset. */
		cudaMemset(gpu_unwrap_buffer_, 0, sizeof(float) * nb_unwrap_elts);
	}

	UnwrappingResources::~UnwrappingResources()
	{
		if (gpu_unwrap_buffer_)
			cudaFree(gpu_unwrap_buffer_);
		if (gpu_predecessor_)
			cudaFree(gpu_predecessor_);
		if (gpu_angle_predecessor_)
			cudaFree(gpu_angle_predecessor_);
		if (gpu_angle_current_)
			cudaFree(gpu_angle_current_);
		if (gpu_angle_copy_)
			cudaFree(gpu_angle_copy_);
		if (gpu_unwrapped_angle_)
			cudaFree(gpu_unwrapped_angle_);
	}

	bool UnwrappingResources::cudaRealloc(void *ptr, const size_t size)
	{
		if (ptr)
			cudaFree(ptr);
		if (cudaMalloc(&ptr, size) != cudaSuccess)
			return (false);
		return (true);
	}

	void UnwrappingResources::reallocate(const size_t image_size)
	{
		int err = 0;
		// We compare requested memory against available memory, and reallocate if needed.
		if (capacity_ <= total_memory_)
			return;

		total_memory_ = capacity_;
		auto nb_unwrap_elts = image_size * capacity_;

		err |= cudaRealloc(gpu_unwrap_buffer_, sizeof(float) * nb_unwrap_elts);
		err |= cudaRealloc(gpu_predecessor_, sizeof(cufftComplex) * image_size);
		err |= cudaRealloc(gpu_angle_predecessor_, sizeof(float) * image_size);
		err |= cudaRealloc(gpu_angle_current_, sizeof(float) * image_size);
		err |= cudaRealloc(gpu_angle_copy_, sizeof(float) * image_size);
		err |= cudaRealloc(gpu_unwrapped_angle_, sizeof(float) * image_size);
		if (err != 0)
			throw std::exception("Cannot reallocate UnwrappingResources");
		cudaMemset(gpu_unwrap_buffer_, 0, sizeof(float) * nb_unwrap_elts);
	}

	void UnwrappingResources::reset(const size_t capacity)
	{
		capacity_ = capacity;
		size_ = 0;
		next_index_ = 0;
	}
}