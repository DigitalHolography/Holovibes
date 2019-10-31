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

/*! \file
 *
 * Nppi function wrappers.
 * Functions are templated then specialized because nppi functions have a
 * different name for each type. */

#pragma once

#include <nppi.h>

#include "nppi_data.hh"
#include "unique_ptr.hh"
#include "logger.hh"
#include "tools.hh"

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

			// nppi_get_min
#pragma region NPPI_GET_MIN

			// The generic version should not be called
			// Use the sepcialized versions
			template <typename T>
			NppStatus nppi_get_min_(T* image, NppiData& nppi_data, T* min)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_get_min_<float>(float* image, NppiData& nppi_data, float* min)
			{
				return nppiMin_32f_C1R(image,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					nppi_data.get_scratch_buffer(&nppiMaxGetBufferHostSize_32f_C1R),
					min);
			}
#pragma endregion NPPI_GET_MIN

			// nppi_get_min_index
#pragma region NPPI_GET_MIN_INDEX

			// The generic version should not be called
			// Use the sepcialized versions
			template <typename T>
			NppStatus nppi_get_min_index_(T* image, NppiData& nppi_data, T* min, int* min_x, int* min_y)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_get_min_index_<float>(float* image, NppiData& nppi_data, float* min, int* min_x, int* min_y)
			{
				return nppiMinIndx_32f_C1R(image,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					nppi_data.get_scratch_buffer(&nppiMaxIndxGetBufferHostSize_32f_C1R),
					min, min_x, min_y);
			}
#pragma endregion NPPI_GET_MIN_INDEX

			// nppi_get_min_max
#pragma region NPPI_GET_MIN_MAX

			// The generic version should not be called
			// Use the sepcialized versions
			template <typename T>
			NppStatus nppi_get_min_max_(T* image, NppiData& nppi_data, T* min, T* max)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_get_min_max_<float>(float* image, NppiData& nppi_data, float* min, float* max)
			{
				return nppiMinMax_32f_C1R(image,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					min,
					max,
					nppi_data.get_scratch_buffer(&nppiMinMaxGetBufferHostSize_32f_C1R));
			}
#pragma endregion NPPI_GET_MIN_MAX

			// nppi_get_min_max_index
#pragma region NPPI_GET_MIN_MAX_INDEX

			// The generic version should not be called
			// Use the sepcialized versions
			template <typename T>
			NppStatus nppi_get_min_max_index_(T* image, NppiData& nppi_data, T* min, NppiPoint* min_idx, T* max, NppiPoint* max_idx)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_get_min_max_index_<float>(float* image, NppiData& nppi_data, float* min, NppiPoint* min_idx, float* max, NppiPoint* max_idx)
			{
				return nppiMinMaxIndx_32f_C1R(image,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					min,
					max,
					min_idx,
					max_idx,
					nppi_data.get_scratch_buffer(&nppiMinMaxIndxGetBufferHostSize_32f_C1R));
			}
#pragma endregion NPPI_GET_MIN_MAX_INDEX

			// nppi_add_constant
#pragma region NPPI_ADD_CONSTANT

			template <typename T>
			NppStatus nppi_add_constant_(T* src, NppiData& nppi_data, T constant)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_add_constant_<float>(float* image, NppiData& nppi_data, float constant)
			{
				return nppiAddC_32f_C1IR(constant, image, nppi_data.get_step<float>(), nppi_data.get_size());
			}

			template<>
			NppStatus nppi_add_constant_<cuComplex>(cuComplex* image, NppiData& nppi_data, cuComplex constant)
			{
				return nppiAddC_32fc_C1IR(Npp32fc{ constant.x, constant.y },
					(Npp32fc*)image,
					nppi_data.get_step<cuComplex>(),
					nppi_data.get_size());
			}
#pragma endregion NPPI_ADD_CONSTANT

			// nppi_add
#pragma region NPPI_ADD

			template <typename T>
			NppStatus nppi_add_(T* img1, T* img2, NppiData& nppi_data)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_add_<float>(float* img1, float* img2, NppiData& nppi_data)
			{
				return nppiAdd_32f_C1IR(img2,
					nppi_data.get_step<float>(),
					img1,
					nppi_data.get_step<float>(),
					nppi_data.get_size());
			}

			template<>
			NppStatus nppi_add_<cuComplex>(cuComplex* img1, cuComplex* img2, NppiData& nppi_data)
			{
				return nppiAdd_32fc_C1IR((Npp32fc*)img2,
					nppi_data.get_step<cuComplex>(),
					(Npp32fc*)img1,
					nppi_data.get_step<cuComplex>(),
					nppi_data.get_size());
			}
#pragma endregion NPPI_ADD

			// nppi_sub_constant
#pragma region NPPI_SUB_CONSTANT

			template <typename T>
			NppStatus nppi_sub_constant_(T* src, NppiData& nppi_data, T constant)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_sub_constant_<float>(float* image, NppiData& nppi_data, float constant)
			{
				return nppiSubC_32f_C1IR(constant, image, nppi_data.get_step<float>(), nppi_data.get_size());
			}

			template<>
			NppStatus nppi_sub_constant_<cuComplex>(cuComplex* image, NppiData& nppi_data, cuComplex constant)
			{
				return nppiSubC_32fc_C1IR(Npp32fc{ constant.x, constant.y },
					(Npp32fc*)image,
					nppi_data.get_step<cuComplex>(),
					nppi_data.get_size());
			}
#pragma endregion NPPI_SUB_CONSTANT

			// nppi_sub
#pragma region NPPI_SUB

			template <typename T>
			NppStatus nppi_sub_(T* img1, T* img2, NppiData& nppi_data)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_sub_<float>(float* img1, float* img2, NppiData& nppi_data)
			{
				return nppiSub_32f_C1IR(img2,
					nppi_data.get_step<float>(),
					img1,
					nppi_data.get_step<float>(),
					nppi_data.get_size());
			}

			template<>
			NppStatus nppi_sub_<cuComplex>(cuComplex* img1, cuComplex* img2, NppiData& nppi_data)
			{
				return nppiSub_32fc_C1IR((Npp32fc*)img2,
					nppi_data.get_step<cuComplex>(),
					(Npp32fc*)img1,
					nppi_data.get_step<cuComplex>(),
					nppi_data.get_size());
			}
#pragma endregion NPPI_SUB

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

			// nppi_divide
#pragma region NPPI_DIVIDE

			template <typename T>
			NppStatus nppi_divide_(T* img1, T* img2, NppiData& nppi_data)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_divide_<float>(float* img1, float* img2, NppiData& nppi_data)
			{
				return nppiDiv_32f_C1IR(img2,
					nppi_data.get_step<float>(),
					img1,
					nppi_data.get_step<float>(),
					nppi_data.get_size());
			}

			template<>
			NppStatus nppi_divide_<cuComplex>(cuComplex* img1, cuComplex* img2, NppiData& nppi_data)
			{
				return nppiDiv_32fc_C1IR((Npp32fc*)img2,
					nppi_data.get_step<cuComplex>(),
					(Npp32fc*)img1,
					nppi_data.get_step<cuComplex>(),
					nppi_data.get_size());
			}
#pragma endregion NPPI_DIVIDE

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

			template<>
			NppStatus nppi_multiply_by_constant_<cuComplex>(cuComplex* image, NppiData& nppi_data, cuComplex constant)
			{
				return nppiMulC_32fc_C1IR(Npp32fc{ constant.x, constant.y },
					(Npp32fc*)image,
					nppi_data.get_step<cuComplex>(),
					nppi_data.get_size());
			}
#pragma endregion NPPI_MULTIPLY_BY_CONSTANT

			// nppi_multiply
#pragma region NPPI_MULTIPLY

			template <typename T>
			NppStatus nppi_multiply_(T* img1, T* img2, NppiData& nppi_data)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_multiply_<float>(float* img1, float* img2, NppiData& nppi_data)
			{
				return nppiMul_32f_C1IR(img2,
					nppi_data.get_step<float>(),
					img1,
					nppi_data.get_step<float>(),
					nppi_data.get_size());
			}

			template<>
			NppStatus nppi_multiply_<cuComplex>(cuComplex* img1, cuComplex* img2, NppiData& nppi_data)
			{
				return nppiMul_32fc_C1IR((Npp32fc*)img2,
					nppi_data.get_step<cuComplex>(),
					(Npp32fc*)img1,
					nppi_data.get_step<cuComplex>(),
					nppi_data.get_size());
			}
#pragma endregion NPPI_MULTIPLY

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
				NppiPoint offset{ 0, 0 };
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

			// nppi_convolution
#pragma region NPPI_CONVOLUTION

			template <typename T>
			NppStatus nppi_convolution_(T* src, T* dst, NppiData& nppi_data, T* kernel, NppiData& nppi_kernel_data)
			{
				static_assert(false);
			}

			template<>
			NppStatus nppi_convolution_<float>(float* src, float* dst, NppiData& nppi_data, float* kernel, NppiData& nppi_kernel_data)
			{
				NppiPoint anchor{ nppi_kernel_data.get_size().width / 2, nppi_kernel_data.get_size().height / 2 };
				NppiPoint offset{ 0, 0 };
				return nppiFilterBorder_32f_C1R(src,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					offset,
					dst,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					kernel,
					nppi_kernel_data.get_size(),
					anchor,
					NPP_BORDER_REPLICATE);
			}
#pragma endregion NPPI_CONVOLUTION

		} // anonymous namespace

		/*! Gets the max value in the image
		* \param image The image
		* \param nppi_data NppiData corresponding to the image
		* \param max Result for the max pixel value
		*/
		template <typename T>
		NppStatus nppi_get_max(T* image, NppiData& nppi_data, T* max)
		{
			UniquePtr<T> max_gpu(1);

			NppStatus ret = nppi_get_max_(image, nppi_data, max_gpu.get());

			cudaMemcpy(max, max_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);

			return ret;
		}

		/*! Gets the max value and its index in the image
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

		/*! Gets the min value in the image
		* \param image The image
		* \param nppi_data NppiData corresponding to the image
		* \param min Result for the min pixel value
		*/
		template <typename T>
		NppStatus nppi_get_min(T* image, NppiData& nppi_data, T* min)
		{
			UniquePtr<T> min_gpu(1);

			NppStatus ret = nppi_get_min_(image, nppi_data, min_gpu.get());

			cudaMemcpy(min, min_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);

			return ret;
		}

		/*! Gets the min value and its index in the image
		* \param image The image
		* \param nppi_data NppiData corresponding to the image
		* \param min Result for the min pixel value (can be null if you only need the index)
		* \param min_x Result for the X coordinate of the min pixel
		* \param min_y Result for the Y coordinate of the min pixel
		*/
		template <typename T>
		NppStatus nppi_get_min_index(T* image, NppiData& nppi_data, T* min, int* min_x, int* min_y)
		{
			UniquePtr<T> min_gpu(1);
			UniquePtr<int> min_index_gpu(2);

			NppStatus ret = nppi_get_min_index_(image, nppi_data, min_gpu.get(), min_index_gpu.get(), min_index_gpu.get() + 1);

			if (min != nullptr)
				cudaMemcpy(min, min_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(min_x, min_index_gpu.get(), sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(min_y, min_index_gpu.get() + 1, sizeof(int), cudaMemcpyDeviceToHost);

			return ret;
		}

		/*! Gets the min and max value in the image
		* \param image The image
		* \param nppi_data NppiData corresponding to the image
		* \param min Result for the min pixel value
		* \param max Result for the max pixel value
		*/
		template <typename T>
		NppStatus nppi_get_min_max(T* image, NppiData& nppi_data, T* min, T* max)
		{
			UniquePtr<T> min_gpu(1);
			UniquePtr<T> max_gpu(1);

			NppStatus ret = nppi_get_min_max_<float>(image, nppi_data, min_gpu.get(), max_gpu.get());

			cudaMemcpy(min, min_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(max, max_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);

			return ret;
		}

		/*! Gets the min & max value and their indices in the image
		* \param image The image
		* \param nppi_data NppiData corresponding to the image
		* \param min Result for the min pixel value (can be null if you only need the index)
		* \param min_x Result for the X coordinate of the min pixel
		* \param min_y Result for the Y coordinate of the min pixel
		* \param max Result for the max pixel value (can be null if you only need the index)
		* \param max_x Result for the X coordinate of the max pixel
		* \param max_y Result for the Y coordinate of the max pixel
		*/
		template <typename T>
		NppStatus nppi_get_min_max_index(T* image, NppiData& nppi_data, T* min, int* min_x, int* min_y, T* max, int* max_x, int* max_y)
		{
			UniquePtr<T> min_gpu(1);
			UniquePtr<NppiPoint> min_index_gpu(1);

			UniquePtr<T> max_gpu(1);
			UniquePtr<NppiPoint> max_index_gpu(1);

			NppStatus ret = nppi_get_min_max_index_(image,
				nppi_data,
				min_gpu.get(),
				min_index_gpu.get(),
				max_gpu.get(),
				max_index_gpu.get());

			if (min != nullptr)
				cudaMemcpy(min, min_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);
			int min_index_buf[2];
			cudaMemcpy(min_index_buf, min_index_gpu.get(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
			*min_x = min_index_buf[0];
			*min_y = min_index_buf[1];

			if (max != nullptr)
				cudaMemcpy(max, max_gpu.get(), sizeof(T), cudaMemcpyDeviceToHost);
			int max_index_buf[2];
			cudaMemcpy(max_index_buf, max_index_gpu.get(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
			*max_x = max_index_buf[0];
			*max_y = max_index_buf[1];

			return ret;
		}

		/*! Adds a constant to every pixel of an image
		* \param image The source image
		* \param nppi_data NppiData corresponding to the image
		* \param constant The value to add
		*/
		template <typename T>
		NppStatus nppi_add_constant(T* image, NppiData& nppi_data, T constant)
		{
			NppStatus ret = nppi_add_constant_(image, nppi_data, constant);

			return ret;
		}

		/*! Adds 2 images pixel by pixel
		* \param img1 Source AND destination image
		* \param img2 Source image
		* \param nppi_data NppiData corresponding to the images
		*/
		template <typename T>
		NppStatus nppi_add(T* img1, T* img2, NppiData& nppi_data)
		{
			NppStatus ret = nppi_add_(img1, img2, nppi_data);

			return ret;
		}

		/*! Subtracts a constant to every pixel of an image
		* \param image The source image
		* \param nppi_data NppiData corresponding to the image
		* \param constant The value to subtract
		*/
		template <typename T>
		NppStatus nppi_sub_constant(T* image, NppiData& nppi_data, T constant)
		{
			NppStatus ret = nppi_sub_constant_(image, nppi_data, constant);

			return ret;
		}

		/*! Subtracts 2 images pixel by pixel
		* \param img1 Source AND destination image
		* \param img2 Source image
		* \param nppi_data NppiData corresponding to the images
		*/
		template <typename T>
		NppStatus nppi_sub(T* img1, T* img2, NppiData& nppi_data)
		{
			NppStatus ret = nppi_sub_(img1, img2, nppi_data);

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

		/*! Divides 2 images pixel by pixel
		* \param img1 Source AND destination image
		* \param img2 Source image
		* \param nppi_data NppiData corresponding to the images
		*/
		template <typename T>
		NppStatus nppi_divide(T* img1, T* img2, NppiData& nppi_data)
		{
			NppStatus ret = nppi_divide_(img1, img2, nppi_data);

			return ret;
		}

		/*! Multiplies every pixel of an image by a constant
		* \param image The source image
		* \param nppi_data NppiData corresponding to the image
		* \param constant The multiplication factor
		*/
		template <typename T>
		NppStatus nppi_multiply_by_constant(T* image, NppiData& nppi_data, T constant)
		{
			NppStatus ret = nppi_multiply_by_constant_(image, nppi_data, constant);

			return ret;
		}

		/*! Multiplies 2 images pixel by pixel
		* \param img1 Source AND destination image
		* \param img2 Source image
		* \param nppi_data NppiData corresponding to the images
		*/
		template <typename T>
		NppStatus nppi_multiply(T* img1, T* img2, NppiData& nppi_data)
		{
			NppStatus ret = nppi_multiply_(img1, img2, nppi_data);

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

		/*! Apply a high pass filter on an image
		* \param src The source image
		* \param dst The destination image (will allocate memory if src == dst)
		* \param nppi_data NppiData corresponding to the images
		* \param mask_size The mask going to be applied
		*/
		template <typename T>
		NppStatus nppi_high_pass_filter(T* src, T* dst, NppiData& nppi_data, NppiMaskSize mask_size)
		{
			T* tmp_dst = dst;
			size_t size = nppi_data.get_size().width * nppi_data.get_size().height;
			if (src == dst)
			{
				T* tmp;
				cudaMalloc(&tmp, size * sizeof(T));
				tmp_dst = tmp;
			}

			NppStatus ret = nppi_high_pass_filter_(src, tmp_dst, nppi_data, mask_size);

			if (src == dst)
			{
				cudaMemcpy(src, tmp_dst, size * sizeof(T), cudaMemcpyDeviceToDevice);
				cudaFree(tmp_dst);
			}

			return ret;
		}

		/*! Computes the convolution of an image and a kernel
		* \param src The source image
		* \param dst The destination image (will allocate memory if src == dst)
		* \param nppi_data The NppiData conrresponding to src and dst
		* \param kernel The convolution kernel to apply
		* \param nppi_kernel_data The NppiData corresponding to the kernel
		*/
		template<typename T>
		NppStatus nppi_convolution(T* src, T* dst, NppiData& nppi_data, T* kernel, NppiData& nppi_kernel_data)
		{
			T* tmp_dst = dst;
			size_t size = nppi_data.get_size().width * nppi_data.get_size().height;
			if (src == dst)
			{
				T* tmp;
				cudaMalloc(&tmp, size * sizeof(T));
				tmp_dst = tmp;
			}

			NppStatus ret = nppi_convolution_(src, tmp_dst, nppi_data, kernel, nppi_kernel_data);

			if (src == dst)
			{
				cudaMemcpy(src, tmp_dst, size * sizeof(T), cudaMemcpyDeviceToDevice);
				cudaFree(tmp_dst);
			}

			return ret;
		}

		template<typename T>
		void nppi_normalize(T* src, NppiData& nppi_data)
		{
			double mean = 0;
			nppi_mean(src, nppi_data, &mean);
			nppi_divide_by_constant(src, nppi_data, static_cast<float>(mean));
			nppi_multiply_by_constant(src, nppi_data, 65535.0f);
		}
	} // namespace cuda_tools
} // namespace holovibes