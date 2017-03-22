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

#include "queue.hh"
#include "tools_conversion.cuh"

#include "info_manager.hh"
#include <stdexcept>

namespace holovibes
{

	Queue::Queue(const camera::FrameDescriptor& frame_desc, const unsigned int elts, std::string name)
		: frame_desc_(frame_desc)
		, size_(frame_desc_.frame_size())
		, pixels_(frame_desc_.frame_res())
		, max_elts_(elts)
		, curr_elts_(0)
		, start_(0)
		, display_(true)
		, is_big_endian_(frame_desc.depth >= 2 &&
			frame_desc.endianness == camera::BIG_ENDIAN)
		, name_(name)
		, buffer_(nullptr)
	{
		if (cudaMalloc(&buffer_, size_ * elts) != CUDA_SUCCESS)
		{
			std::cerr << "Queue: couldn't allocate queue" << std::endl;
			std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
			throw std::logic_error(name_ + ": couldn't allocate queue");
		}
		frame_desc_.endianness = camera::LITTLE_ENDIAN;
		cudaStreamCreate(&stream_);
	}

	Queue::~Queue()
	{
		if (display_)
			gui::InfoManager::remove_info(name_);
		if (buffer_)
			if (cudaFree(buffer_) != CUDA_SUCCESS)
				std::cerr << "Queue: couldn't free queue" << std::endl;
		cudaStreamDestroy(stream_);
	}

	size_t Queue::get_size() const
	{
		return size_;
	}

	void* Queue::get_buffer()
	{
		return buffer_;
	}

	const camera::FrameDescriptor& Queue::get_frame_desc() const
	{
		return frame_desc_;
	}

	int Queue::get_pixels()
	{
		return pixels_;
	}

	unsigned int Queue::get_max_elts() const
	{
		return max_elts_;
	}

	void* Queue::get_start()
	{
		std::lock_guard<std::mutex> Guard(mutex_);
		return buffer_ + start_ * size_;
	}

	unsigned int Queue::get_start_index()
	{
		std::lock_guard<std::mutex> Guard(mutex_);
		return start_;
	}

	void* Queue::get_end()
	{
		std::lock_guard<std::mutex> Guard(mutex_);
		return buffer_ + ((start_ + curr_elts_) % max_elts_) * size_;
	}

	void* Queue::get_last_images(const unsigned n)
	{
		std::lock_guard<std::mutex> Guard(mutex_);
		return buffer_ + ((start_ + curr_elts_ - n) % max_elts_) * size_;
	}

	unsigned int Queue::get_end_index()
	{
		std::lock_guard<std::mutex> Guard(mutex_);
		return (start_ + curr_elts_) % max_elts_;
	}

	bool Queue::enqueue(void* elt, cudaMemcpyKind cuda_kind)
	{
		std::lock_guard<std::mutex> Guard(mutex_);

		const unsigned int end_ = (start_ + curr_elts_) % max_elts_;
		char* new_elt_adress = buffer_ + (end_ * size_);
		cudaError_t cuda_status = cudaMemcpyAsync(new_elt_adress,
			elt,
			size_,
			cuda_kind,
			stream_);
		if (cuda_status != CUDA_SUCCESS)
		{
 			std::cerr << "Queue: couldn't enqueue" << std::endl;
			std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
			if (display_)
				gui::InfoManager::update_info(name_, "couldn't enqueue");
			if (buffer_)
			{
				cudaFree(buffer_);
				buffer_ = nullptr;
			}
			return false;
		}
		if (is_big_endian_)
			endianness_conversion(
				reinterpret_cast<unsigned short *>(new_elt_adress),
				reinterpret_cast<unsigned short *>(new_elt_adress),
				frame_desc_.frame_res(), stream_);

		if (curr_elts_ < max_elts_)
			++curr_elts_;
		else
			start_ = (start_ + 1) % max_elts_;
		if (display_)
			gui::InfoManager::update_info(name_,
				std::to_string(curr_elts_) + std::string("/") + std::to_string(max_elts_)
				+ std::string(" (") + calculate_size() + std::string(" MB)"));
		return true;
	}

	void Queue::dequeue(void* dest, cudaMemcpyKind cuda_kind)
	{
		if (curr_elts_ > 0)
		{
			void* first_img = buffer_ + start_ * size_;
			cudaMemcpyAsync(dest, first_img, size_, cuda_kind, stream_);
			start_ = (start_ + 1) % max_elts_;
			--curr_elts_;
			if (display_)
				gui::InfoManager::update_info(name_,
					std::to_string(curr_elts_) + std::string("/") + std::to_string(max_elts_)
					+ std::string(" (") + calculate_size() + std::string(" MB)"));
		}
	}

	void Queue::dequeue()
	{
		std::lock_guard<std::mutex> Guard(mutex_);

		if (curr_elts_ > 0)
		{
			start_ = (start_ + 1) % max_elts_;
			--curr_elts_;
		}
	}

	void Queue::flush()
	{
		std::lock_guard<std::mutex> Guard(mutex_);

		curr_elts_ = 0;
		start_ = 0;
	}

	void Queue::set_display(bool value)
	{
		display_ = value;
	}

	std::string Queue::calculate_size(void)
	{
		std::string display_size = std::to_string((get_max_elts() * get_size()) >> 20); // get_size() / (1024 * 1024)
		size_t pos = display_size.find(".");

		if (pos != std::string::npos)
			display_size.resize(pos);
		return(display_size);
	}

	std::mutex& Queue::getGuard()
	{
		return (mutex_);
	}

}