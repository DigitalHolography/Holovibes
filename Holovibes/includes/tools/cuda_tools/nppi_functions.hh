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

#include <nppi.h>

#include "nppi_data.hh"
#include "unique_ptr.hh"

namespace holovibes
{
	namespace cuda_tools
	{
		//
		// get max
		//

		// The generic version should not be called
		// Use the sepcialized versions
		template <typename T>
		void nppi_get_max_(T* image, NppiData& nppi_data, T* max)
		{
			static_assert(false);
		}

		template<>
		void nppi_get_max_<float>(float* image, NppiData& nppi_data, float* max)
		{
			nppiMax_32f_C1R(image,
				nppi_data.get_step<float>(),
				nppi_data.get_size(),
				nppi_data.get_scratch_buffer(&nppiMaxGetBufferHostSize_32f_C1R),
				max);
		}

		template <typename T>
		void nppi_get_max(T* image, NppiData& nppi_data, T* max)
		{
			UniquePtr<T> max_gpu(1);

			nppi_get_max_<float>(image, nppi_data, max_gpu.get());

			cudaMemcpy(&max, max_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);
		}

		//
		// get max index
		//

		// The generic version should not be called
		// Use the sepcialized versions
		template <typename T>
		static void nppi_get_max_index_(T* image, NppiData& nppi_data, T* max, int* max_x, int* max_y)
		{
			static_assert(false);
		}

		template<>
		static void nppi_get_max_index_<float>(float* image, NppiData& nppi_data, float* max, int* max_x, int* max_y)
		{
			nppiMaxIndx_32f_C1R(image,
				nppi_data.get_step<float>(),
				nppi_data.get_size(),
				nppi_data.get_scratch_buffer(&nppiMaxIndxGetBufferHostSize_32f_C1R),
				max, max_x, max_y);
		}

		template <typename T>
		void nppi_get_max_index(T* image, NppiData& nppi_data, T* max, int* max_x, int* max_y)
		{
			UniquePtr<T> max_gpu(1);
			UniquePtr<int> max_index_gpu(2);

			nppi_get_max_index_(image, nppi_data, max_gpu.get(), max_index_gpu.get(), max_index_gpu.get() + 1);

			cudaMemcpy(max, max_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(max_x, max_index_gpu.get(), sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(max_y, max_index_gpu.get() + 1, sizeof(int), cudaMemcpyDeviceToHost);
		}

	} // namespace cuda_tools
} // namespace holovibes