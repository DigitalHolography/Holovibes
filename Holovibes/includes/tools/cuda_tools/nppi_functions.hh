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

			// nppi_divide_by_constant
#pragma region NPPI_DIVIDE_BY_CONSTANT

			template <typename T>
			NppStatus nppi_divide_by_constant_(T* src, NppiData& nppi_data, T constant)
			{
				std::cerr << "Cannot call non specialized template function" << std::endl;
				abort();
			}

			template<>
			NppStatus nppi_divide_by_constant_<float>(float* image, NppiData& nppi_data, float constant)
			{
				if (nppi_data.get_num_channels() == 3)
				{
					Npp32f constants[3] = { constant, constant, constant };
					return nppiDivC_32f_C3IR(constants, image, 3 * nppi_data.get_step<float>(), nppi_data.get_size());
				}
				
				return nppiDivC_32f_C1IR(constant, image, nppi_data.get_step<float>(), nppi_data.get_size());
			}

			template<>
			NppStatus nppi_divide_by_constant_<cuComplex>(cuComplex* image, NppiData& nppi_data, cuComplex constant)
			{
				return nppiDivC_32fc_C1IR(*((Npp32fc*)&constant), (Npp32fc*)image, nppi_data.get_step<cuComplex>(), nppi_data.get_size());
			}
#pragma endregion NPPI_DIVIDE_BY_CONSTANT

// nppi_multiply_by_constant
#pragma region NPPI_MULTIPLY_BY_CONSTANT

			template <typename T>
			NppStatus nppi_multiply_by_constant_(T* src, NppiData& nppi_data, T constant)
			{
				std::cerr << "Cannot call non specialized template function" << std::endl;
				abort();
			}

			template<>
			NppStatus nppi_multiply_by_constant_<float>(float* image, NppiData& nppi_data, float constant)
			{
				if (nppi_data.get_num_channels() == 3)
				{
					Npp32f constants[3] = { constant, constant, constant };
					return nppiMulC_32f_C3IR(constants, image, 3 * nppi_data.get_step<float>(), nppi_data.get_size());
				}
				
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

			// nppi_mean
#pragma region NPPI_MEAN

			template <typename T>
			NppStatus nppi_mean_(T* src, NppiData& nppi_data, double* mean)
			{
				std::cerr << "Cannot call non specialized template function" << std::endl;
				abort();
			}

			template<>
			NppStatus nppi_mean_<float>(float* image, NppiData& nppi_data, double* mean)
			{
				if (nppi_data.get_num_channels() == 3)
					return nppiMean_32f_C3R(image,
						3 * nppi_data.get_step<float>(),
						nppi_data.get_size(),
						nppi_data.get_scratch_buffer(&nppiMeanGetBufferHostSize_32f_C3R),
						mean);

				return nppiMean_32f_C1R(image,
					nppi_data.get_step<float>(),
					nppi_data.get_size(),
					nppi_data.get_scratch_buffer(&nppiMeanGetBufferHostSize_32f_C1R),
					mean);
			}
#pragma endregion NPPI_MEAN
		} // anonymous namespace


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
		* \param constant The multiplication factor
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

		template<typename T>
		void nppi_normalize(T* src, NppiData& nppi_data, unsigned mult_constant)
		{
			double mean = 0;
			nppi_mean(src, nppi_data, &mean);
			nppi_divide_by_constant(src, nppi_data, static_cast<float>(mean));
			nppi_multiply_by_constant(src, nppi_data, static_cast<float>(1 << mult_constant));
		}
	} // namespace cuda_tools
} // namespace holovibes