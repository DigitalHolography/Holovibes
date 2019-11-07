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

#include <cuda.h>

#include "queue.hh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools.cuh"


#include "info_manager.hh"
#include "logger.hh"

namespace holovibes
{
	using camera::FrameDescriptor;
	using camera::Endianness;

	using MutexGuard = std::lock_guard<std::mutex>;

	Queue::Queue(const camera::FrameDescriptor& frame_desc, const unsigned int elts, std::string name, unsigned int input_width, unsigned int input_height, unsigned int elm_size)
		: frame_desc_(frame_desc)
		, frame_size_(frame_desc_.frame_size())
		, frame_resolution_(frame_desc_.frame_res())
		, max_elts_(elts)
		, curr_elts_(0)
		, start_index_(0)
		, is_big_endian_(frame_desc.depth >= 2 &&
			frame_desc.byteEndian == Endianness::BigEndian)
		, name_(name)
		, data_buffer_()
		, stream_()
		, display_(true)
		, input_width_(input_width)
		, input_height_(input_height)
		, elm_size_(elm_size)
		, square_input_mode_(SquareInputMode::NO_MODIFICATION)
	{
		if (!elts || !data_buffer_.resize(frame_size_ * elts))
		{
			LOG_ERROR("Queue: couldn't allocate queue");
			throw std::logic_error(name_ + ": couldn't allocate queue");
		}

		//Needed if input is embedded into a bigger square
		cudaMemset(data_buffer_.get(), 0, frame_size_ * elts);

		cudaCheckError();

		frame_desc_.byteEndian = Endianness::LittleEndian;
		cudaStreamCreate(&stream_);
	}

	Queue::~Queue()
	{
		if (display_)
			gui::InfoManager::get_manager()->remove_info(name_);
		cudaStreamDestroy(stream_);
	}

	void Queue::resize(const unsigned int size)
	{
		max_elts_ = size;

		if (!max_elts_ || !data_buffer_.resize(frame_size_ * max_elts_))
		{
			LOG_ERROR("Queue: couldn't resize queue");
			throw std::logic_error(name_ + ": couldn't resize queue");
		}

		//Needed if input is embedded into a bigger square
		cudaMemset(data_buffer_.get(), 0, frame_size_ * max_elts_);

		cudaCheckError();

		curr_elts_ = 0;
		start_index_ = 0;
	}

	size_t Queue::get_size() const
	{
		return frame_size_;
	}

	void* Queue::get_buffer()
	{
		return data_buffer_;
	}

	const camera::FrameDescriptor& Queue::get_frame_desc() const
	{
		return frame_desc_;
	}

	int Queue::get_pixels()
	{
		return frame_resolution_;
	}

	unsigned int Queue::get_max_elts() const
	{
		return max_elts_;
	}

	void* Queue::get_start()
	{
		return data_buffer_.get() + start_index_ * frame_size_;
	}

	unsigned int Queue::get_start_index()
	{
		return start_index_;
	}

	void* Queue::get_end()
	{
		return data_buffer_.get() + ((start_index_ + curr_elts_) % max_elts_) * frame_size_;
	}

	void* Queue::get_last_images(const unsigned n)
	{
		return data_buffer_.get() + ((start_index_ + curr_elts_ - n) % max_elts_) * frame_size_;
	}

	unsigned int Queue::get_end_index()
	{
		return (start_index_ + curr_elts_) % max_elts_;
	}

	bool Queue::enqueue(void* elt, cudaMemcpyKind cuda_kind)
	{
		MutexGuard mGuard(mutex_);

		const uint	end_ = (start_index_ + curr_elts_) % max_elts_;
		char		*new_elt_adress = data_buffer_.get() + (end_ * frame_size_);

		cudaError_t cuda_status;
		switch (square_input_mode_)
		{
			case SquareInputMode::NO_MODIFICATION:
				cuda_status = cudaMemcpyAsync(new_elt_adress,
											  elt,
											  frame_size_,
				 							  cuda_kind,
											  stream_);
				break;
			case SquareInputMode::ZERO_PADDED_SQUARE:
				//The black bands should have been written at the allocation of the data buffer
				cuda_status = embed_into_square(static_cast<char *>(elt),
												input_width_,
												input_height_,
												new_elt_adress,
												elm_size_,
												cuda_kind,
												stream_); 
				break;
			case SquareInputMode::CROPPED_SQUARE:
				cuda_status = crop_into_square(static_cast<char *>(elt),
											   input_width_,
											   input_height_,
											   new_elt_adress,
											   elm_size_,
											   cuda_kind,
											   stream_);
				break;
			default:
				assert(false);
				LOG_ERROR(std::string("Missing switch case for square input mode"));
				if (display_)
					gui::InfoManager::get_manager()->update_info(name_, "couldn't enqueue");
				return false;
		}

		if (cuda_status != CUDA_SUCCESS)
		{
			LOG_ERROR(std::string("Queue: couldn't enqueue into ") + std::string(name_) + std::string(": ") + std::string(cudaGetErrorString(cuda_status)));
			if (display_)
				gui::InfoManager::get_manager()->update_info(name_, "couldn't enqueue");
			data_buffer_.reset();
			return false;
		}
		if (is_big_endian_)
			endianness_conversion(
				reinterpret_cast<ushort *>(new_elt_adress),
				reinterpret_cast<ushort *>(new_elt_adress),
				frame_desc_.frame_res(), stream_);

		if (curr_elts_ < max_elts_)
			++curr_elts_;
		else
			start_index_ = (start_index_ + 1) % max_elts_;
		if (display_)
			display_queue_to_InfoManager();
		return true;
	}

	void Queue::dequeue(void* dest, cudaMemcpyKind cuda_kind)
	{
		if (curr_elts_ > 0)
		{
			void* first_img = data_buffer_.get() + start_index_ * frame_size_;
			cudaMemcpyAsync(dest, first_img, frame_size_, cuda_kind, stream_);
			start_index_ = (start_index_ + 1) % max_elts_;
			--curr_elts_;
			if (display_)
				display_queue_to_InfoManager();
		}
	}

	void Queue::dequeue_48bit_to_24bit(void * dest, cudaMemcpyKind cuda_kind)
	{
		if (curr_elts_ > 0)
		{
			void* first_img = data_buffer_.get() + start_index_ * frame_size_;
			cuda_tools::UniquePtr<uchar> tmp_uchar(frame_size_ / 2);
			ushort_to_uchar(static_cast<ushort*>(first_img), tmp_uchar, frame_resolution_ * 3);
			cudaMemcpy(dest, tmp_uchar, frame_resolution_ * 3, cuda_kind);
			start_index_ = (start_index_ + 1) % max_elts_;
			--curr_elts_;
			if (display_)
				display_queue_to_InfoManager();
		}
	}

	void Queue::dequeue()
	{
		MutexGuard mGuard(mutex_);

		if (curr_elts_ > 0)
		{
			start_index_ = (start_index_ + 1) % max_elts_;
			--curr_elts_;
		}
	}

	void Queue::flush()
	{
		MutexGuard mGuard(mutex_);

		curr_elts_ = 0;
		start_index_ = 0;
	}

	void Queue::set_display(bool value)
	{
		display_ = value;
	}

	void Queue::set_square_input_mode(SquareInputMode mode)
	{
		square_input_mode_ = mode;
	}

	void Queue::display_queue_to_InfoManager() const
	{
		std::string message = std::to_string(curr_elts_) + "/" + std::to_string(max_elts_) + " (" + calculate_size() + " MB)";

		if (name_ == "InputQueue")
			gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::INPUT_QUEUE, name_, message);
		else if (name_ == "OutputQueue")
			gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::OUTPUT_QUEUE, name_, message);
		else if (name_ == "STFTQueue")
			gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::STFT_QUEUE, name_, message);
		else if (name_ == "RawOutputQueue")
			gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::RAW_OUTPUT_QUEUE, name_, message);
	}

	std::string Queue::calculate_size(void) const
	{
		std::string display_size = std::to_string((get_max_elts() * get_size()) >> 20); // get_size() / (1024 * 1024)
		size_t pos = display_size.find(".");

		if (pos != std::string::npos)
			display_size.resize(pos);
		return display_size;
	}

	std::mutex& Queue::getGuard()
	{
		return mutex_;
	}

}