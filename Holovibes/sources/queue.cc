#include <cuda.h>

#include "queue.hh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools.cuh"
#include "cuda_memory.cuh"
#include "frame_reshape.cuh"

#include "logger.hh"
#include "holovibes.hh"

namespace holovibes
{
using camera::Endianness;
using camera::FrameDescriptor;

Queue::Queue(const camera::FrameDescriptor& fd,
             const unsigned int max_size,
             QueueType type,
             unsigned int input_width,
             unsigned int input_height,
             unsigned int bytes_per_pixel)
    : DisplayQueue(fd)
    , entry_(GSH::fast_updates_map<QueueType>.create_entry(type, true))
    , size_(entry_->first)
    , max_size_(entry_->second)
    , type_(type)
    , start_index_(0)
    , is_big_endian_(fd.depth >= 2 && fd.byteEndian == Endianness::BigEndian)
    , input_width_(input_width)
    , input_height_(input_height)
    , bytes_per_pixel(bytes_per_pixel)
    , has_overridden_(false)
{
    max_size_ = max_size;
    size_ = 0;

    if (max_size_ == 0 || !data_.resize(fd_.frame_size() * max_size_))
    {
        LOG_ERROR << "Queue: could not allocate queue";
        throw std::logic_error("Could not allocate queue");
    }

    // Needed if input is embedded into a bigger square
    cudaXMemset(data_.get(), 0, fd_.frame_size() * max_size_);

    fd_.byteEndian = Endianness::LittleEndian;
}

Queue::~Queue() { GSH::fast_updates_map<QueueType>.remove_entry(type_); }

void Queue::resize(const unsigned int size, const cudaStream_t stream)
{
    MutexGuard mGuard(mutex_);

    max_size_ = size;

    if (max_size_ == 0 || !data_.resize(fd_.frame_size() * max_size_))
    {
        LOG_ERROR << "Queue: could not resize queue";
        throw std::logic_error("Could not resize queue");
    }

    // Needed if input is embedded into a bigger square
    cudaXMemsetAsync(data_.get(), 0, fd_.frame_size() * max_size_, stream);
    cudaXStreamSynchronize(stream);

    size_ = 0;
    start_index_ = 0;
}

bool Queue::enqueue(void* elt, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
{
    MutexGuard mGuard(mutex_);

    const uint end_ = (start_index_ + size_) % max_size_;
    char* new_elt_adress = data_.get() + (end_ * fd_.frame_size());

    cudaError_t cuda_status;
    // No async needed for Qt buffer
    cuda_status = cudaMemcpyAsync(new_elt_adress, elt, fd_.frame_size(), cuda_kind, stream);

    if (cuda_status != CUDA_SUCCESS)
    {
        LOG_ERROR << "Queue: could not enqueue: " << std::string(cudaGetErrorString(cuda_status));
        data_.reset();
        return false;
    }

    if (is_big_endian_)
        endianness_conversion(reinterpret_cast<ushort*>(new_elt_adress),
                              reinterpret_cast<ushort*>(new_elt_adress),
                              1,
                              static_cast<uint>(fd_.frame_res()),
                              stream);

    // Synchronize after the copy has been lauched and before updating the size
    cudaXStreamSynchronize(stream);

    if (size_ < max_size_)
        ++size_;
    else
    {
        start_index_ = (start_index_ + 1) % max_size_;
        has_overridden_ = true;
    }

    return true;
}

void Queue::enqueue_multiple_aux(
    void* out, void* in, unsigned int nb_elts, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
{
    cudaXMemcpyAsync(out, in, nb_elts * fd_.frame_size(), cuda_kind, stream);

    if (is_big_endian_)
        endianness_conversion(reinterpret_cast<ushort*>(out),
                              reinterpret_cast<ushort*>(out),
                              nb_elts,
                              fd_.frame_res(),
                              stream);
}

void Queue::copy_multiple(Queue& dest, unsigned int nb_elts, const cudaStream_t stream)
{
    MutexGuard m_guard_src(mutex_);
    MutexGuard m_guard_dst(dest.get_guard());

    if (nb_elts > size_)
        nb_elts = size_;

    unsigned int tmp_src_start_index = start_index_;
    if (nb_elts > dest.max_size_)
    {
        start_index_ = (start_index_ + nb_elts - dest.max_size_) % max_size_;
        nb_elts = dest.max_size_;
    }

    // Determine regions info
    struct QueueRegion src;
    if (start_index_ + nb_elts > max_size_)
    {
        src.first = static_cast<char*>(get_start());
        src.first_size = max_size_ - start_index_;
        src.second = data_.get();
        src.second_size = nb_elts - src.first_size;
    }
    else
    {
        src.first = static_cast<char*>(get_start());
        src.first_size = nb_elts;
    }

    struct QueueRegion dst;
    const uint begin_to_enqueue_index = (dest.start_index_ + dest.size_) % dest.max_size_;
    char* begin_to_enqueue = dest.data_.get() + (begin_to_enqueue_index * dest.fd_.frame_size());
    if (begin_to_enqueue_index + nb_elts > dest.max_size_)
    {
        dst.first = begin_to_enqueue;
        dst.first_size = dest.max_size_ - begin_to_enqueue_index;
        dst.second = dest.data_.get();
        dst.second_size = nb_elts - dst.first_size;
    }
    else
    {
        dst.first = begin_to_enqueue;
        dst.first_size = nb_elts;
    }

    copy_multiple_aux(src, dst, static_cast<uint>(fd_.frame_size()), stream);

    // Synchronize after every copy has been lauched and before updating the
    // size
    cudaXStreamSynchronize(stream);

    // Update dest queue parameters
    dest.size_ += nb_elts;
    if (dest.size_ > dest.max_size_)
    {
        dest.start_index_ = (dest.start_index_ + dest.size_) % dest.max_size_;
        dest.size_.store(dest.max_size_.load());
        dest.has_overridden_ = true;
    }

    start_index_ = tmp_src_start_index;
}

void Queue::copy_multiple_aux(QueueRegion& src, QueueRegion& dst, const uint frame_size, const cudaStream_t stream)
{
    // Handle copies depending on regions info
    if (src.overflow())
    {
        if (dst.overflow())
        {
            if (src.first_size > dst.first_size)
            {
                cudaXMemcpyAsync(dst.first, src.first, dst.first_size * frame_size, cudaMemcpyDeviceToDevice, stream);
                src.consume_first(dst.first_size, frame_size);

                cudaXMemcpyAsync(dst.second, src.first, src.first_size * frame_size, cudaMemcpyDeviceToDevice, stream);
                dst.consume_second(src.first_size, frame_size);

                cudaXMemcpyAsync(dst.second,
                                 src.second,
                                 src.second_size * frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 stream);
            }
            else // src.first_size <= dst.first_size
            {
                cudaXMemcpyAsync(dst.first, src.first, src.first_size * frame_size, cudaMemcpyDeviceToDevice, stream);
                dst.consume_first(src.first_size, frame_size);

                if (src.second_size > dst.first_size)
                {
                    cudaXMemcpyAsync(dst.first,
                                     src.second,
                                     dst.first_size * frame_size,
                                     cudaMemcpyDeviceToDevice,
                                     stream);
                    src.consume_second(dst.first_size, frame_size);

                    cudaXMemcpyAsync(dst.second,
                                     src.second,
                                     src.second_size * frame_size,
                                     cudaMemcpyDeviceToDevice,
                                     stream);
                }
                else // src.second_size == dst.first_size
                {
                    cudaXMemcpyAsync(dst.first,
                                     src.second,
                                     src.second_size * frame_size,
                                     cudaMemcpyDeviceToDevice,
                                     stream);
                }
            }
        }
        else
        {
            // In this case: dst.first_size > src.first_size

            cudaXMemcpyAsync(dst.first, src.first, src.first_size * frame_size, cudaMemcpyDeviceToDevice, stream);
            dst.consume_first(src.first_size, frame_size);

            cudaXMemcpyAsync(dst.first, src.second, dst.first_size * frame_size, cudaMemcpyDeviceToDevice, stream);
        }
    }
    else
    {
        if (dst.overflow())
        {
            // In this case: src.first_size > dst.first_size

            cudaXMemcpyAsync(dst.first, src.first, dst.first_size * frame_size, cudaMemcpyDeviceToDevice, stream);
            src.consume_first(dst.first_size, frame_size);

            cudaXMemcpyAsync(dst.second, src.first, src.first_size * frame_size, cudaMemcpyDeviceToDevice, stream);
        }
        else
        {
            cudaXMemcpyAsync(dst.first, src.first, src.first_size * frame_size, cudaMemcpyDeviceToDevice, stream);
        }
    }
}

bool Queue::enqueue_multiple(void* elts, unsigned int nb_elts, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
{
    MutexGuard mGuard(mutex_);

    // To avoid templating the Queue
    char* elts_char = static_cast<char*>(elts);
    if (nb_elts > max_size_)
    {
        elts_char = elts_char + nb_elts * fd_.frame_size() - max_size_ * fd_.frame_size();
        // skip overwritten elts
        start_index_ = (start_index_ + nb_elts - max_size_) % max_size_;
        nb_elts = max_size_;
    }

    const uint begin_to_enqueue_index = (start_index_ + size_) % max_size_;
    void* begin_to_enqueue = data_.get() + (begin_to_enqueue_index * fd_.frame_size());

    if (begin_to_enqueue_index + nb_elts > max_size_)
    {
        unsigned int nb_elts_to_insert_at_end = max_size_ - begin_to_enqueue_index;
        enqueue_multiple_aux(begin_to_enqueue, elts_char, nb_elts_to_insert_at_end, stream, cuda_kind);

        elts_char += nb_elts_to_insert_at_end * fd_.frame_size();

        unsigned int nb_elts_to_insert_at_beginning = nb_elts - nb_elts_to_insert_at_end;
        enqueue_multiple_aux(data_.get(), elts_char, nb_elts_to_insert_at_beginning, stream, cuda_kind);
    }
    else
    {
        enqueue_multiple_aux(begin_to_enqueue, elts_char, nb_elts, stream, cuda_kind);
    }

    cudaXStreamSynchronize(stream);

    size_ += nb_elts;
    // Overwrite older elements in the queue -> update start_index
    if (size_ > max_size_)
    {
        start_index_ = (start_index_ + size_ - max_size_) % max_size_;
        size_.store(max_size_.load());
        has_overridden_ = true;
    }

    return true;
}

void Queue::enqueue_from_48bit(void* src, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
{
    cuda_tools::UniquePtr<uchar> src_uchar(fd_.frame_size());
    ushort_to_uchar(static_cast<ushort*>(src), src_uchar, static_cast<uint>(fd_.frame_size()), stream);
    enqueue(src_uchar, stream, cuda_kind);
}

void Queue::dequeue(void* dest, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
{
    MutexGuard mGuard(mutex_);

    CHECK(size_ > 0) << "Queue size cannot be empty at dequeue";
    void* first_img = data_.get() + start_index_ * fd_.frame_size();
    cudaXMemcpyAsync(dest, first_img, fd_.frame_size(), cuda_kind, stream);

    cudaXStreamSynchronize(stream);

    dequeue_non_mutex(); // Update indexes
}

void Queue::dequeue(const unsigned int nb_elts)
{
    MutexGuard mGuard(mutex_);

    dequeue_non_mutex(nb_elts);
}

void Queue::dequeue_non_mutex(const unsigned int nb_elts)
{
    CHECK(size_ >= nb_elts) << "When dequeuing " << nb_elts << " elements, queue size should be bigger than it, not "
                            << size_;
    size_ -= nb_elts;
    start_index_ = (start_index_ + nb_elts) % max_size_;
}

void Queue::clear()
{
    MutexGuard mGuard(mutex_);

    size_ = 0;
    start_index_ = 0;
}

bool Queue::is_full() const { return max_size_ == size_; }

std::string Queue::calculate_size(void) const
{
    std::string display_size = std::to_string((get_max_size() * fd_.frame_res()) >> 20); // get_size() / (1024 * 1024)
    size_t pos = display_size.find(".");

    if (pos != std::string::npos)
        display_size.resize(pos);
    return display_size;
}
} // namespace holovibes
