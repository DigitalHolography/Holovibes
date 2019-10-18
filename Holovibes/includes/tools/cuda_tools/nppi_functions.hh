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
#include "logger.hh"

namespace holovibes
{
	namespace cuda_tools
	{
		namespace
		{
			// nppi_get_max
#pragma region NPPI_GET_MAX

			// The generic version should not be called
			// Use the sepcialized versions
			template <typename T>
			NppStatus nppi_get_max_(T* image, NppiData& nppi_data, T* max)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_get_max_<float>(float* image, NppiData& nppi_data, float* max)
			{
				return nppiMax_32f_C1R(image,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					nppi_data.get_scratch_buffer(&nppiMaxGetBufferHostSize_32f_C1R),
					max);
			}
#pragma endregion NPPI_GET_MAX

			// nppi_get_max_index
#pragma region NPPI_GET_MAX_INDEX

			// The generic version should not be called
			// Use the sepcialized versions
			template <typename T>
			NppStatus nppi_get_max_index_(T* image, NppiData& nppi_data, T* max, int* max_x, int* max_y)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_get_max_index_<float>(float* image, NppiData& nppi_data, float* max, int* max_x, int* max_y)
			{
				return nppiMaxIndx_32f_C1R(image,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					nppi_data.get_scratch_buffer(&nppiMaxIndxGetBufferHostSize_32f_C1R),
					max, max_x, max_y);
			}
#pragma endregion NPPI_GET_MAX_INDEX

			// nppi_high_pass_filter
#pragma region NPPI_HIGH_PASS_FILTER

			template <typename T>
			NppStatus nppi_high_pass_filter_(T* src, T* dst, NppiData& nppi_data, NppiMaskSize mask_size)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_high_pass_filter_<float>(float* src, float* dst, NppiData& nppi_data, NppiMaskSize mask_size)
			{
				NppiPoint offset{0, 0};
				return nppiFilterHighPassBorder_32f_C1R(src,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					offset,
					dst,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					mask_size,
					NppiBorderType::NPP_BORDER_REPLICATE);
			}
#pragma endregion NPPI_HIGH_PASS_FILTER

			// nppi_divide_by_constant
#pragma region NPPI_DIVIDE_BY_CONSTANT

			template <typename T>
			NppStatus nppi_divide_by_constant_(T* src, NppiData& nppi_data, T constant)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_divide_by_constant_<float>(float* image, NppiData& nppi_data, float constant)
			{
				return nppiDivC_32f_C1IR(constant, image, nppi_data.get_step<float>(), nppi_data.get_size());
			}
#pragma endregion NPPI_DIVIDE_BY_CONSTANT

			// nppi_multiply_by_constant
#pragma region NPPI_MULTIPLY_BY_CONSTANT

			template <typename T>
			NppStatus nppi_multiply_by_constant_(T* src, NppiData& nppi_data, T constant)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_multiply_by_constant_<float>(float* image, NppiData& nppi_data, float constant)
			{
				return nppiMulC_32f_C1IR(constant, image, nppi_data.get_step<float>(), nppi_data.get_size());
			}
#pragma endregion NPPI_MULTIPLY_BY_CONSTANT

			// nppi_mean
#pragma region NPPI_MEAN

			template <typename T>
			NppStatus nppi_mean_(T* src, NppiData& nppi_data, double* mean)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_mean_<float>(float* image, NppiData& nppi_data, double* mean)
			{
				return nppiMean_32f_C1R(image,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					nppi_data.get_scratch_buffer(&nppiMeanGetBufferHostSize_32f_C1R),
					mean);
			}
#pragma endregion NPPI_MEAN

		}

		/*! Get the max value in the image
		* \param image The image
		* \param nppi_data NppiData corresponding to the image
		* \param max Result for the max pixel value (can be null if you only need the index)
		*/
		template <typename T>
		NppStatus nppi_get_max(T* image, NppiData& nppi_data, T* max)
		{
			UniquePtr<T> max_gpu(1);

			NppStatus ret = nppi_get_max_<float>(image, nppi_data, max_gpu.get());

			cudaMemcpy(max, max_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);

			return ret;
		}

		/*! Get the max value and its index in the image
		* \param image The image
		* \param nppi_data NppiData corresponding to the image
		* \param max Result for the max pixel value (can be null if you only need the index)
		* \param max_x Result for the X coordinate of the max pixel
		* \param max_y Result for the Y coordinate of the max pixel
		*/
		template <typename T>
		NppStatus nppi_get_max_index(T* image, NppiData& nppi_data, T* max, int* max_x, int* max_y)
		{
			UniquePtr<T> max_gpu(1);
			UniquePtr<int> max_index_gpu(2);

			NppStatus ret = nppi_get_max_index_(image, nppi_data, max_gpu.get(), max_index_gpu.get(), max_index_gpu.get() + 1);

			if (max != nullptr)
				cudaMemcpy(max, max_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(max_x, max_index_gpu.get(), sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(max_y, max_index_gpu.get() + 1, sizeof(int), cudaMemcpyDeviceToHost);

			return ret;
		}

		/*! Apply a high pass filter on an image
		* \param src The source image
		* \param dst The destination image (src != dst)
		* \param nppi_data NppiData corresponding to the images
		* \param mask_size The mask going to be applied
		*/
		template <typename T>
		NppStatus nppi_high_pass_filter(T* src, T* dst, NppiData& nppi_data, NppiMaskSize mask_size)
		{
			if (src == dst)
			{
				LOG_ERROR("could not apply high pass filter, src == dst");
				return NPP_ERROR;
			}

			NppStatus ret = nppi_high_pass_filter_(src, dst, nppi_data, mask_size);

			return ret;
		}

		/*! Divides every pixel of an image by a constant
		* \param image The source image
		* \param nppi_data NppiData corresponding to the image
		* \param constant The divisor
		*/
		template <typename T>
		NppStatus nppi_divide_by_constant(T* image, NppiData& nppi_data, T constant)
		{
			NppStatus ret = nppi_divide_by_constant_(image, nppi_data, constant);

			return ret;
		}

		/*! Multiplies every pixel of an image by a constant
		* \param image The source image
		* \param nppi_data NppiData corresponding to the image
		* \param constant The divisor
		*/
		template <typename T>
		NppStatus nppi_multiply_by_constant(T* image, NppiData& nppi_data, T constant)
		{
			NppStatus ret = nppi_multiply_by_constant_(image, nppi_data, constant);

			return ret;
		}

		/*! Computes the mean (average) of an image
		* \param image The source image
		* \param nppi_data NppiData corresponding to the image
		* \param mean The result mean
		*/
		template <typename T>
		NppStatus nppi_mean(T* image, NppiData& nppi_data, double* mean)
		{
			UniquePtr<double> mean_gpu(1);
			NppStatus ret = nppi_mean_(image, nppi_data, mean_gpu.get());

			cudaMemcpy(mean, mean_gpu.get(), sizeof(double), cudaMemcpyDeviceToHost);

			return ret;
		}
	} // namespace cuda_tools
} // namespace holovibes