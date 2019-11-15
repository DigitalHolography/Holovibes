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
 * Wrapper around Nppi data required for each call to a Nppi function
 * https://docs.nvidia.com/cuda/npp/index.html
 * https://docs.nvidia.com/cuda/npp/general_conventions_lb.html
 * https://docs.nvidia.com/cuda/npp/nppi_conventions_lb.html
 * https://docs.nvidia.com/cuda/npp/modules.html
 */

#pragma once

#include <nppi.h>
#include <functional>

#include "unique_ptr.hh"

namespace holovibes
{
	namespace cuda_tools
	{
		class NppiData
		{
		public:
			/*! Creates a new NppiData object with
			* size_.width = width
			* size_.height = height */
			NppiData(int width, int height);

			/*! Returns the NppiSize field of the object */
			const NppiSize& get_size() const;

			/*! Sets the NppiSize field to new values
			* \param width New width of the NppiSize field
			* \param height New height of the NppiSize field*/
			void set_size(int width, int height);

			/*! Returns the image line step (number of bytes in 1 line) */
			template <typename T>
			int get_step() const
			{
				return size_.width * sizeof(T);
			}

			/*! Returns the scratch buffer allocated with the required size or nullptr on error
			* \param size_function Nppi function used to get the required size of the scratch buffer */
			Npp8u* get_scratch_buffer(std::function<NppStatus(NppiSize, int*)> size_function);

			/*! Returns the current scratch buffer or nullptr on error */
			static Npp8u* get_scratch_buffer();

			/*! Returns the scratch buffer allocated with the required size or nullptr on error
			* \param size Required size of the scratch buffer */
			static Npp8u* get_scratch_buffer(int size);

		private:
			/*! Size of the image associated with this NppiData object */
			NppiSize size_;

			/*! Current size of the scratch buffer */
			static int scratch_buffer_size_;
			/*! Scratch buffer used by Nppi function. Used for every calls to Nppi functions.
			* Reallocated when the required size is greater than the current size; */
			static UniquePtr<Npp8u> scratch_buffer_;
		};
	} // namepsace cuda_tools
} // namespace holovibes