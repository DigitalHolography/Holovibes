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

	inline bool Queue::has_overridden() const
	{
		return has_overridden_;
	}
}
