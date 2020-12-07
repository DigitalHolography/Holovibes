#include "queue.hh"

namespace holovibes
{
    inline size_t Queue::get_frame_size() const
    {
        return frame_size_;
    }

    inline void* Queue::get_data() const
	{
		return data_;
	}

	inline const camera::FrameDescriptor& Queue::get_fd() const
	{
		return fd_;
	}

	inline size_t Queue::get_frame_res() const
	{
		return frame_res_;
	}

    inline unsigned int Queue::get_size() const
    {
        return size_;
    }

    inline unsigned int Queue::get_max_size() const
	{
		return max_size_;
	}

	inline void* Queue::get_start() const
	{
		return data_.get() + start_index_ * frame_size_;
	}

	inline unsigned int Queue::get_start_index() const
	{
		return start_index_;
	}

	inline void* Queue::get_end() const
	{
		return data_.get() + ((start_index_ + size_) % max_size_) * frame_size_;
	}

	inline void* Queue::get_last_image() const
	{
		MutexGuard mGuard(mutex_);
		// if the queue is empty, return a random frame
		return data_.get() + ((start_index_ + size_ - 1) % max_size_) * frame_size_;
	}

    inline unsigned int Queue::get_end_index() const
    {
		return (start_index_ + size_) % max_size_;
    }

	inline void Queue::set_square_input_mode(SquareInputMode mode)
	{
		square_input_mode_ = mode;
	}

	inline std::mutex& Queue::get_guard()
	{
		return mutex_;
	}
}
