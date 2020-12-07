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
#include "cuda_memory.cuh"

#include "logger.hh"
#include "holovibes.hh"

namespace holovibes
{
	using camera::FrameDescriptor;
	using camera::Endianness;

	Queue::Queue(const camera::FrameDescriptor& fd,
				 const unsigned int max_size,
				 Queue::QueueType type,
				 unsigned int input_width,
				 unsigned int input_height,
				 unsigned int bytes_per_pixel)
		: fd_(fd)
		, frame_size_(fd_.frame_size())
		, frame_res_(fd_.frame_res())
		, max_size_(max_size)
		, type_(type)
		, size_(0)
		, start_index_(0)
		, is_big_endian_(fd.depth >= 2 &&
			fd.byteEndian == Endianness::BigEndian)
		, data_()
		, input_width_(input_width)
		, input_height_(input_height)
		, bytes_per_pixel(bytes_per_pixel)
		, square_input_mode_(SquareInputMode::NO_MODIFICATION)
	{
		if (max_size_ == 0 || !data_.resize(frame_size_ * max_size_))
		{
			LOG_ERROR("Queue: could not allocate queue");
			throw std::logic_error("Could not allocate queue");
		}

		// Needed if input is embedded into a bigger square
		cudaXMemset(data_.get(), 0, frame_size_ * max_size_);

		fd_.byteEndian = Endianness::LittleEndian;

		Holovibes::instance().get_info_container().add_queue_size(type_, size_, max_size_);
	}

	Queue::~Queue()
	{
		Holovibes::instance().get_info_container().remove_queue_size(type_);
	}

	void Queue::resize(const unsigned int size)
	{
		MutexGuard mGuard(mutex_);

		max_size_ = size;

		if (max_size_ == 0 || !data_.resize(frame_size_ * max_size_))
		{
			LOG_ERROR("Queue: could not resize queue");
			throw std::logic_error("Could not resize queue");
		}

		//Needed if input is embedded into a bigger square
		cudaXMemset(data_.get(), 0, frame_size_ * max_size_);

		size_ = 0;
		start_index_ = 0;
	}

	bool Queue::enqueue(void* elt, cudaMemcpyKind cuda_kind)
	{
		MutexGuard mGuard(mutex_);

		const uint	end_ = (start_index_ + size_) % max_size_;
		char		*new_elt_adress = data_.get() + (end_ * frame_size_);

		cudaError_t cuda_status;
		switch (square_input_mode_)
		{
			case SquareInputMode::NO_MODIFICATION:
				// No async needed for Qt buffer
				cuda_status = cudaMemcpy(new_elt_adress,
											  elt,
											  frame_size_,
				 							  cuda_kind);
				break;
			case SquareInputMode::ZERO_PADDED_SQUARE:
				//The black bands should have been written at the allocation of the data buffer
				cuda_status = embed_into_square(static_cast<char *>(elt),
												input_width_,
												input_height_,
												new_elt_adress,
												bytes_per_pixel,
												cuda_kind);
				break;
			case SquareInputMode::CROPPED_SQUARE:
				cuda_status = crop_into_square(static_cast<char *>(elt),
											   input_width_,
											   input_height_,
											   new_elt_adress,
											   bytes_per_pixel,
											   cuda_kind);
				break;
			default:
				assert(false);
				LOG_ERROR("Missing switch case for square input mode. Could not enqueue!");
				return false;
		}

		if (cuda_status != CUDA_SUCCESS)
		{
 			LOG_ERROR(std::string("Queue: could not enqueue into: ") + std::string(cudaGetErrorString(cuda_status)));
			data_.reset();
			return false;
		}

		if (is_big_endian_)
			endianness_conversion(
				reinterpret_cast<ushort *>(new_elt_adress),
				reinterpret_cast<ushort *>(new_elt_adress),
				1,
				frame_res_);

		if (size_ < max_size_)
			++size_;
		else
			start_index_ = (start_index_ + 1) % max_size_;

		return true;
	}

	void Queue::enqueue_multiple_aux(void *out,
									 void *in,
									 unsigned int nb_elts,
									 cudaMemcpyKind cuda_kind)
	{
		switch (square_input_mode_)
		{
			case SquareInputMode::NO_MODIFICATION:
				cudaXMemcpyAsync(out, in, nb_elts * frame_size_, cuda_kind);
				break;
			case SquareInputMode::ZERO_PADDED_SQUARE:
				batched_embed_into_square(static_cast<char *>(in),
												input_width_,
												input_height_,
												static_cast<char *>(out),
												bytes_per_pixel,
												nb_elts);
				break;
			case SquareInputMode::CROPPED_SQUARE:
				batched_crop_into_square(static_cast<char *>(in),
											input_width_,
											input_height_,
											static_cast<char *>(out),
											bytes_per_pixel,
											nb_elts);
				break;
			default:
				assert(false);
				LOG_ERROR("Missing switch case for square input mode. Could not enqueue!");
		}

		if (is_big_endian_)
			endianness_conversion(
				reinterpret_cast<ushort *>(out),
				reinterpret_cast<ushort *>(out),
				nb_elts,
				fd_.frame_res());
	}

	void Queue::copy_multiple(Queue& dest, unsigned int nb_elts)
	{
		MutexGuard m_guard_src(mutex_);
		MutexGuard m_guard_dst(dest.get_guard());

		// The buffer of the queues is stored in the device memory
		cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice;

		if (nb_elts > size_)
			nb_elts = size_;

		assert(nb_elts <= dest.max_size_);

		// Determine regions info
		struct QueueRegion src;
		if (start_index_ + nb_elts > max_size_)
		{
			src.first = static_cast<char *>(get_start());
			src.first_size = max_size_ - start_index_;
			src.second = data_.get();
			src.second_size = nb_elts - src.first_size;
		}
		else
		{
			src.first = static_cast<char *>(get_start());
			src.first_size = nb_elts;
		}

		struct QueueRegion dst;
		const uint begin_to_enqueue_index = (dest.start_index_ + dest.size_) % dest.max_size_;
		void *begin_to_enqueue = dest.data_.get() + (begin_to_enqueue_index * dest.frame_size_);
		if (begin_to_enqueue_index + nb_elts > dest.max_size_)
		{
			dst.first = static_cast<char *>(begin_to_enqueue);
			dst.first_size = dest.max_size_ - begin_to_enqueue_index;
			dst.second = dest.data_.get();
			dst.second_size = nb_elts - dst.first_size;
		}
		else
		{
			dst.first = static_cast<char *>(begin_to_enqueue);
			dst.first_size = nb_elts;
		}

		// Handle copies depending on regions info
		if (src.overflow())
		{
			if (dst.overflow())
			{
				if (src.first_size > dst.first_size)
				{
					cudaXMemcpyAsync(dst.first, src.first, dst.first_size * frame_size_, cuda_kind);
					src.consume_first(dst.first_size, frame_size_);

					cudaXMemcpyAsync(dst.second, src.first, src.first_size * frame_size_, cuda_kind);
					dst.consume_second(src.first_size, frame_size_);

					cudaXMemcpyAsync(dst.second, src.second, src.second_size * frame_size_, cuda_kind);
				}
				else // src.first_size <= dst.first_size
				{
					cudaXMemcpyAsync(dst.first, src.first, src.first_size * frame_size_, cuda_kind);
					dst.consume_first(src.first_size, frame_size_);

					if (src.second_size > dst.first_size)
					{
						cudaXMemcpyAsync(dst.first, src.second, dst.first_size * frame_size_, cuda_kind);
						src.consume_second(dst.first_size, frame_size_);

						cudaXMemcpyAsync(dst.second, src.second, src.second_size * frame_size_, cuda_kind);
					}
					else // src.second_size == dst.first_size
					{
						cudaXMemcpyAsync(dst.first, src.second, src.second_size * frame_size_, cuda_kind);
					}
				}
			}
			else
			{
				// In this case: dst.first_size > src.first_size

				cudaXMemcpyAsync(dst.first, src.first, src.first_size * frame_size_, cuda_kind);
				dst.consume_first(src.first_size, frame_size_);

				cudaXMemcpyAsync(dst.first, src.second, dst.first_size * frame_size_, cuda_kind);
			}
		}
		else
		{
			if (dst.overflow())
			{
				// In this case: src.first_size > dst.first_size

				cudaXMemcpyAsync(dst.first, src.first, dst.first_size * frame_size_, cuda_kind);
				src.consume_first(dst.first_size, frame_size_);

				cudaXMemcpyAsync(dst.second, src.first, src.first_size * frame_size_, cuda_kind);
			}
			else
			{
				cudaXMemcpyAsync(dst.first, src.first, src.first_size * frame_size_, cuda_kind);
			}
		}

		// Update dest queue parameters
		dest.size_ += nb_elts;
		if (dest.size_ > dest.max_size_)
		{
			dest.start_index_ = (dest.start_index_ + dest.size_ - dest.max_size_) % dest.max_size_;
			dest.size_.store(dest.max_size_.load());
		}
	}

	bool Queue::enqueue_multiple(void* elts, unsigned int nb_elts, cudaMemcpyKind cuda_kind)
	{
		MutexGuard mGuard(mutex_);

		// To avoid templating the Queue
		char* elts_char = static_cast<char *>(elts);
		if (nb_elts > max_size_)
		{
			elts_char = elts_char + nb_elts * frame_size_ - max_size_ * frame_size_;
			// skip overwritten elts
			start_index_ = (start_index_ + nb_elts - max_size_) % max_size_;
			nb_elts = max_size_;
		}

		const uint begin_to_enqueue_index = (start_index_ + size_) % max_size_;
		void *begin_to_enqueue = data_.get() + (begin_to_enqueue_index * frame_size_);

		if (begin_to_enqueue_index + nb_elts > max_size_)
		{
			unsigned int nb_elts_to_insert_at_end = max_size_ - begin_to_enqueue_index;
			enqueue_multiple_aux(begin_to_enqueue, elts_char, nb_elts_to_insert_at_end, cuda_kind);

			elts_char += nb_elts_to_insert_at_end * frame_size_;

			unsigned int nb_elts_to_insert_at_beginning = nb_elts - nb_elts_to_insert_at_end;
			enqueue_multiple_aux(data_.get(), elts_char, nb_elts_to_insert_at_beginning, cuda_kind);
		}
		else
		{
			enqueue_multiple_aux(begin_to_enqueue, elts_char, nb_elts, cuda_kind);
		}

		size_ += nb_elts;
		// Overwrite older elements in the queue -> update start_index
		if (size_ > max_size_)
		{
			start_index_ = (start_index_ + size_ - max_size_) % max_size_;
			size_.store(max_size_.load());
		}

		return true;
	}

	void Queue::enqueue_from_48bit(void* src, cudaMemcpyKind cuda_kind)
	{
		cuda_tools::UniquePtr<uchar> src_uchar(frame_size_);
		ushort_to_uchar(static_cast<ushort*>(src), src_uchar, frame_size_);
		enqueue(src_uchar, cuda_kind);
	}

	void Queue::dequeue(void* dest, cudaMemcpyKind cuda_kind)
	{
		MutexGuard mGuard(mutex_);

		assert(size_ > 0);
		void* first_img = data_.get() + start_index_ * frame_size_;
		cudaXMemcpyAsync(dest, first_img, frame_size_, cuda_kind);
		dequeue_non_mutex(); // Update indexes

		// If dequeuing in the host side, a device synchronization must be
		// done because the memcpy must be sync in this case
		if (cuda_kind == cudaMemcpyDeviceToHost)
			cudaDeviceSynchronize();
	}

	void Queue::dequeue(const unsigned int nb_elts)
	{
		MutexGuard mGuard(mutex_);

		dequeue_non_mutex(nb_elts);
	}

	void Queue::dequeue_non_mutex(const unsigned int nb_elts)
	{
		assert(size_ >= nb_elts);
		size_ -= nb_elts;
		start_index_ = (start_index_ + nb_elts) % max_size_;
	}

	void Queue::clear()
	{
		MutexGuard mGuard(mutex_);

		size_ = 0;
		start_index_ = 0;
	}

	bool Queue::is_full() const
	{
		return max_size_ == size_;
	}

	std::string Queue::calculate_size(void) const
	{
		std::string display_size = std::to_string((get_max_size() * get_frame_size()) >> 20); // get_size() / (1024 * 1024)
		size_t pos = display_size.find(".");

		if (pos != std::string::npos)
			display_size.resize(pos);
		return display_size;
	}
}