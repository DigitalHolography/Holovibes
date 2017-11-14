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

#include <memory>
#include <functional>
#include <iostream>
#include <cuda_runtime.h>

namespace holovibes
{
	/*! \brief Contains memory handlers for cuda buffers. */
	namespace cuda_tools
	{
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

			/// Allocates an array of size sizeof(T) * size
			UniquePtr(std::size_t size)
				: base(nullptr, cudaFree)
			{
				resize(size);
			}

			/// Allocates an array of size sizeof(T) * size, free the old pointer if not null
			void resize(std::size_t size)
			{
				T* tmp;
				auto code = cudaMalloc(&tmp, size * sizeof(T));
				if (code != cudaSuccess)
				{
					std::cout << "cudaMalloc error:" << cudaGetErrorString(code) << std::endl;
					tmp = nullptr;
				}
				reset(tmp);
			}
		};
	}
}
