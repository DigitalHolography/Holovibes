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
 * cuda_tools::UniquePtr containing an array */
#pragma once

namespace holovibes
{
	namespace cuda_tools
	{
		/// Array class for cuda buffers that ocasionally need to be resized
		template <typename T>
		class Array : public UniquePtr<T>
		{
		public:
			using base = UniquePtr<T>;

			/// Intantiate an empty / nullptr array
			Array()
				: base()
				, size_(0)
			{}

			/// Creates an array of size sizeof(T) * size
			Array(std::size_t size)
				: base(size)
				, size_(size)
			{}

			/// Realloc the array only if needed
			/// 
			/// \return if the resize succeeded
			bool ensure_minimum_size(std::size_t size)
			{
				if (size <= size_)
					return true;
				resize(size);
				if (get())
				{
					size_ = size;
					return true;
				}
				size_ = 0;
				return false;
			}

			/// Is the array size greater or equal to size
			bool is_large_enough(std::size_t size) const
			{
				return size_ >= size;
			}

			/// Resize the array
			void resize(std::size_t size)
			{
				base::resize(size);
				size_ = size;
			}

		private:
			std::size_t size_;
		};
	}
}