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

#include "nppi_data.hh"

namespace holovibes
{
	namespace cuda_tools
	{
		int NppiData::scratch_buffer_size_ = 0;
		UniquePtr<Npp8u> NppiData::scratch_buffer_;

		NppiData::NppiData(int width, int height)
		{
			size_.width = width;
			size_.height = height;
			num_channels_ = 1;
		}

		NppiData::NppiData(int width, int height, unsigned int num_channels)
		{
			size_.width = width;
			size_.height = height;
			num_channels_ = num_channels;
		}

		const NppiSize& NppiData::get_size() const
		{
			return size_;
		}

		void NppiData::set_size(int width, int height)
		{
			size_.width = width;
			size_.height = height;
		}

		Npp8u* NppiData::get_scratch_buffer(std::function<NppStatus(NppiSize, int*)> size_function)
		{
			int required_size = 0;
			if (size_function(size_, &required_size) != NPP_SUCCESS)
				return nullptr;
			return get_scratch_buffer(required_size);
		}

		Npp8u* NppiData::get_scratch_buffer()
		{
			return get_scratch_buffer(scratch_buffer_size_);
		}

		Npp8u* NppiData::get_scratch_buffer(int size)
		{
			if (scratch_buffer_.get() == nullptr || size > scratch_buffer_size_)
			{
				scratch_buffer_size_ = size;
				scratch_buffer_.resize(size);
			}
			return scratch_buffer_.get();
		}

		unsigned int NppiData::get_num_channels() const
		{
			return num_channels_;
		}

	} // namepsace cuda_tools
} // namespace holovibes

