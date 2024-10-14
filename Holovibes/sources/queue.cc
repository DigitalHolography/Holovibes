#include <cuda.h>
#include <spdlog/spdlog.h>

#include "queue.hh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools.cuh"
#include "cuda_memory.cuh"
#include "frame_reshape.cuh"

#include "logger.hh"
#include "holovibes.hh"
#include "API.hh"

namespace holovibes
{
using camera::Endianness;
using camera::FrameDescriptor;

HoloQueue::HoloQueue(QueueType type, const Device device)
    : fast_updates_entry_(GSH::fast_updates_map<QueueType>.create_entry(type, true))
    , type_(type)
    , device_(std::get<2>(*fast_updates_entry_))
    , start_index_(0)
    , has_overwritten_(false)
    , size_(std::get<0>(*fast_updates_entry_))
{
    device_ = device;
    data_ = cuda_tools::UniquePtr<char>(device_);
}

Queue::Queue(const camera::FrameDescriptor& fd, const unsigned int max_size, QueueType type, const Device device)
    : DisplayQueue(fd)
    , HoloQueue(type, device)
    , max_size_(std::get<1>(*fast_updates_entry_)) //(fast_updates_entry_->second)
    , is_big_endian_(fd.depth >= 2 && fd.byteEndian == Endianness::BigEndian)
{
    max_size_ = max_size;
    // Check if we have enough memory to allocate the queue, otherwise reduce the size and relaunch the process.
    if (!data_.resize(fd_.get_frame_size() * max_size_))
    {
        bool is_size_modified = false;
        while (!data_.resize(fd_.get_frame_size() * max_size_))
        {
            max_size_--;
        }
        switch (type)
        {
        case QueueType::INPUT_QUEUE:
            api::set_input_buffer_size(max_size_);
            is_size_modified = true;
            break;
        case QueueType::OUTPUT_QUEUE:
            api::set_output_buffer_size(max_size_);
            is_size_modified = true;
            break;
        case QueueType::RECORD_QUEUE:
            api::set_record_buffer_size(max_size_);
            is_size_modified = true;
            break;
        case QueueType::UNDEFINED:
            break;
        default:
            break;
        }
        if (is_size_modified)
        {
            LOG_WARN("Queue: not enough memory to allocate queue. Queue size was reduced to " +
                     std::to_string(max_size_));
            // LOG_WARN("Queue: not enough memory to allocate queue. Queue size was reduced to {}", max_size_);
            // Return because when we set the buffer_size in the switch, the process is relaaunch and the ctor will be
            // called again
            return;
        }
    }

    if (max_size_ == 0 || !data_.resize(fd_.get_frame_size() * max_size_))
    {
        LOG_ERROR("Queue: could not allocate queue");

        throw std::logic_error(std::string("Could not allocate queue (max_size: ") + std::to_string(max_size_) + ")");
    }

    // // Needed if input is embedded into a bigger square
    // if (device_)
    //     cudaXMemset(data_.get(), 0, fd_.get_frame_size() * max_size_);
    // else
    //     std::memset(data_.get(), 0, fd_.get_frame_size() * max_size_);

    cudaXMemset(data_.get(), 0, fd_.get_frame_size() * max_size_);

    fd_.byteEndian = Endianness::LittleEndian;
}

Queue::~Queue() { GSH::fast_updates_map<QueueType>.remove_entry(type_); }

void Queue::rebuild(const camera::FrameDescriptor& fd,
                    const unsigned int size,
                    const cudaStream_t stream,
                    const Device device)
{
    set_fd(fd);

    if (device_ != device)
    {
        device_ = device;
        data_ = cuda_tools::UniquePtr<char>(device_);
    }

    resize(size, stream);
}

void Queue::resize(const unsigned int size, const cudaStream_t stream)
{
    MutexGuard mGuard(mutex_);

    max_size_ = size;

    if (max_size_ == 0 || !data_.resize(fd_.get_frame_size() * max_size_))
    {
        LOG_ERROR("Queue: could not resize queue");
        throw std::logic_error("Could not resize queue");
    }

    // Needed if input is embedded into a bigger square
    // if (device_) {
    //     cudaXMemsetAsync(data_.get(), 0, fd_.get_frame_size() * max_size_, stream);
    //     cudaXStreamSynchronize(stream);
    // }
    // else
    //     std::memset(data_.get(), 0, fd_.get_frame_size() * max_size_);

    cudaXMemsetAsync(data_.get(), 0, fd_.get_frame_size() * max_size_, stream);
    cudaXStreamSynchronize(stream);

    cudaError_t status = cudaGetLastError();

    size_ = 0;
    start_index_ = 0;
}

void Queue::reset()
{
    dequeue(-1);
    has_overwritten_ = false;
}

bool Queue::enqueue(void* elt, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
{
    MutexGuard mGuard(mutex_);

    const uint end_ = (start_index_ + size_) % max_size_;
    char* new_elt_adress = data_.get() + (end_ * fd_.get_frame_size());

    cudaError_t cuda_status;
    // No async needed for Qt buffer
    cuda_status = cudaMemcpyAsync(new_elt_adress, elt, fd_.get_frame_size(), cuda_kind, stream);
    // cuda_status = cudaMemcpy(new_elt_adress, elt, fd_.get_frame_size(), cuda_kind);

    if (cuda_status) // 0 = CUDA_SUCCESS
    {
        LOG_ERROR("Queue: could not enqueue: {}", std::string(cudaGetErrorString(cuda_status)));
        data_.reset();
        return false;
    }

    if (is_big_endian_)
    {
        endianness_conversion(reinterpret_cast<ushort*>(new_elt_adress),
                              reinterpret_cast<ushort*>(new_elt_adress),
                              1,
                              static_cast<uint>(fd_.get_frame_res()),
                              stream);
    }

    // Synchronize after the copy has been lauched and before updating the size
    cudaXStreamSynchronize(stream);

    if (size_ < max_size_)
        ++size_;
    else
    {
        start_index_ = (start_index_ + 1) % max_size_;
        has_overwritten_ = true;
    }

    return true;
}

void Queue::enqueue_multiple_aux(
    void* out, void* in, unsigned int nb_elts, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
{
    cudaXMemcpyAsync(out, in, nb_elts * fd_.get_frame_size(), cuda_kind, stream);

    if (is_big_endian_)
        endianness_conversion(reinterpret_cast<ushort*>(out),
                              reinterpret_cast<ushort*>(out),
                              nb_elts,
                              fd_.get_frame_res(),
                              stream);
}

void Queue::copy_multiple(Queue& dest, unsigned int nb_elts, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
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
    char* begin_to_enqueue = dest.data_.get() + (begin_to_enqueue_index * dest.fd_.get_frame_size());
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

    copy_multiple_aux(src, dst, static_cast<uint>(fd_.get_frame_size()), stream, cuda_kind);

    // Synchronize after every copy has been lauched and before updating the
    // size
    cudaXStreamSynchronize(stream);

    // Update dest queue parameters
    dest.size_ += nb_elts;
    if (dest.size_ > dest.max_size_)
    {
        dest.start_index_ = (dest.start_index_ + dest.size_) % dest.max_size_;
        dest.size_.store(dest.max_size_.load());
        dest.has_overwritten_ = true;
    }

    start_index_ = tmp_src_start_index;
}

void Queue::copy_multiple_aux(
    QueueRegion& src, QueueRegion& dst, const size_t frame_size, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
{
    // Handle copies depending on regions info
    if (src.overflow())
    {
        if (dst.overflow())
        {
            if (src.first_size > dst.first_size)
            {
                cudaXMemcpyAsync(dst.first, src.first, dst.first_size * frame_size, cuda_kind, stream);
                src.consume_first(dst.first_size, frame_size);

                cudaXMemcpyAsync(dst.second, src.first, src.first_size * frame_size, cuda_kind, stream);
                dst.consume_second(src.first_size, frame_size);

                cudaXMemcpyAsync(dst.second, src.second, src.second_size * frame_size, cuda_kind, stream);
            }
            else // src.first_size <= dst.first_size
            {
                cudaXMemcpyAsync(dst.first, src.first, src.first_size * frame_size, cuda_kind, stream);
                dst.consume_first(src.first_size, frame_size);

                if (src.second_size > dst.first_size)
                {
                    cudaXMemcpyAsync(dst.first, src.second, dst.first_size * frame_size, cuda_kind, stream);
                    src.consume_second(dst.first_size, frame_size);

                    cudaXMemcpyAsync(dst.second, src.second, src.second_size * frame_size, cuda_kind, stream);
                }
                else // src.second_size == dst.first_size
                {
                    cudaXMemcpyAsync(dst.first, src.second, src.second_size * frame_size, cuda_kind, stream);
                }
            }
        }
        else
        {
            // In this case: dst.first_size > src.first_size

            cudaXMemcpyAsync(dst.first, src.first, src.first_size * frame_size, cuda_kind, stream);
            dst.consume_first(src.first_size, frame_size);

            cudaXMemcpyAsync(dst.first, src.second, dst.first_size * frame_size, cuda_kind, stream);
        }
    }
    else
    {
        if (dst.overflow())
        {
            // In this case: src.first_size > dst.first_size

            cudaXMemcpyAsync(dst.first, src.first, dst.first_size * frame_size, cuda_kind, stream);
            src.consume_first(dst.first_size, frame_size);

            cudaXMemcpyAsync(dst.second, src.first, src.first_size * frame_size, cuda_kind, stream);
        }
        else
        {
            cudaXMemcpyAsync(dst.first, src.first, src.first_size * frame_size, cuda_kind, stream);
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
        elts_char = elts_char + nb_elts * fd_.get_frame_size() - max_size_ * fd_.get_frame_size();
        // skip overwritten elts
        start_index_ = (start_index_ + nb_elts - max_size_) % max_size_;
        nb_elts = max_size_;
    }

    const uint begin_to_enqueue_index = (start_index_ + size_) % max_size_;
    void* begin_to_enqueue = data_.get() + (begin_to_enqueue_index * fd_.get_frame_size());

    if (begin_to_enqueue_index + nb_elts > max_size_)
    {
        unsigned int nb_elts_to_insert_at_end = max_size_ - begin_to_enqueue_index;
        enqueue_multiple_aux(begin_to_enqueue, elts_char, nb_elts_to_insert_at_end, stream, cuda_kind);

        elts_char += nb_elts_to_insert_at_end * fd_.get_frame_size();

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
        has_overwritten_ = true;
    }

    return true;
}

void Queue::enqueue_from_48bit(void* src, const cudaStream_t stream, cudaMemcpyKind cuda_kind)
{
    cuda_tools::CudaUniquePtr<uchar> src_uchar(fd_.get_frame_size());
    ushort_to_uchar(src_uchar, static_cast<ushort*>(src), static_cast<uint>(fd_.get_frame_size()), stream);
    enqueue(src_uchar, stream, cuda_kind);
}

int Queue::dequeue(void* dest, const cudaStream_t stream, cudaMemcpyKind cuda_kind, int nb_elts)
{
    MutexGuard mGuard(mutex_);

    CHECK(size_ > 0, "Queue size cannot be empty at dequeue");
    CHECK(nb_elts >= -1, "Nb elmts must be equal or greater than -1");
    if (nb_elts == -1)
        nb_elts = size_;

    CHECK(std::cmp_less_equal(nb_elts, size_.load()),
          "Request to dequeue {} elts, but the queue has only {}",
          (char)nb_elts,
          (char)size_);

    void* first_img = data_.get() + start_index_ * fd_.get_frame_size();
    cudaXMemcpyAsync(dest, first_img, nb_elts * fd_.get_frame_size(), cuda_kind, stream);

    cudaXStreamSynchronize(stream);

    dequeue_non_mutex(nb_elts); // Update indexes

    return nb_elts;
}

void Queue::dequeue(int nb_elts)
{
    MutexGuard mGuard(mutex_);

    if (nb_elts == -1)
        nb_elts = size_;

    dequeue_non_mutex(nb_elts);
}

void Queue::dequeue_non_mutex(const unsigned int nb_elts)
{
    // CHECK(size_ >= nb_elts, "When dequeuing {} elements, queue size should be bigger than it, not {};", nb_elts,
    // size_);
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
    std::string display_size =
        std::to_string((get_max_size() * fd_.get_frame_res()) >> 20); // get_size() / (1024 * 1024)
    size_t pos = display_size.find(".");

    if (pos != std::string::npos)
        display_size.resize(pos);
    return display_size;
}
} // namespace holovibes
