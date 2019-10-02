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
 * std::unique_ptr "specialization" for cudaFree */
#pragma once

#include <functional>
#include <memory>

#include <cstddef>

#include "logger.hh"

namespace holovibes
{
	/*! \brief Contains memory handlers for cuda buffers. */
	namespace cuda_tools
	{
		namespace _private
		{
			template <typename T>
			struct element_size
			{
				static const std::size_t value = sizeof(T);
			};

			template<>
			struct element_size<void>
			{
				static const std::size_t value = 1;
			};
		}

		/// A smart pointer made for ressources that need to be cudaFreed
		template<typename T>
		class UniquePtr : public std::unique_ptr<T, std::function<void(T*)>>
		{
		public:
			using base = std::unique_ptr<T, std::function<void(T*)>>;
			UniquePtr()
				: base(nullptr, cudaFree)
			{}

			UniquePtr(T *ptr)
				: base(ptr, cudaFree)
			{}

			/// Implicit cast operator
			operator T*()
			{
				return get();
			}

			/// Implicit cast operator
			operator T*() const
			{
				return get();
			}

			/// Allocates an array of size sizeof(T) * size
			UniquePtr(std::size_t size)
				: base(nullptr, cudaFree)
			{
				resize(size);
			}

			/// Allocates an array of size sizeof(T) * size, free the old pointer if not null
			bool resize(std::size_t size)
			{
				T* tmp;
				size *= _private::element_size<T>::value;
				auto code = cudaMalloc(&tmp, size);
				if (code != cudaSuccess)
				{
					LOG_ERROR(std::string("cudaMalloc: ") + cudaGetErrorString(code));
					tmp = nullptr;
				}
				reset(tmp);
				return tmp;
			}
		};
	}
}
