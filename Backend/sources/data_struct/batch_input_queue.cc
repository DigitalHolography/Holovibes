#include <cassert>
#include "logger.hh"
#include "holovibes.hh"

#include "batch_input_queue.hh"
#include "id_queue.hh"
#include "stamp_queue.hh"
#include "API.hh"
#include "enum_record_mode.hh"

extern IdQueue g_record_id_queue;
extern StampQueue g_record_stamp_queue;

namespace holovibes
{
BatchInputQueue::BatchInputQueue(const uint total_nb_frames,
                                 const uint batch_size,
                                 const camera::FrameDescriptor& fd,
                                 const Device device)
    : DisplayQueue(fd)
    , fast_updates_entry_(FastUpdatesMap::map<QueueType>.create_entry(QueueType::INPUT_QUEUE))
    , curr_nb_frames_(std::get<0>(*fast_updates_entry_))  //->first)
    , total_nb_frames_(std::get<1>(*fast_updates_entry_)) //->second)
    , frame_capacity_(total_nb_frames)
    , device_(std::get<2>(*fast_updates_entry_))
{
    device_ = device;
    data_ = cuda_tools::UniquePtr<char>(device_);

    curr_nb_frames_ = 0;
    total_nb_frames_ = total_nb_frames;

    // Set priority of streams
    // Set batch_size and max_size
    create_queue(batch_size);
}

BatchInputQueue::~BatchInputQueue()
{
    destroy_mutexes_streams();
    // data is free as it is a CudaUniquePtr.

    FastUpdatesMap::map<QueueType>.remove_entry(QueueType::INPUT_QUEUE);
}

void BatchInputQueue::create_queue(const uint new_batch_size)
{
    CHECK(new_batch_size > 0, "Batch size cannot be 0.");

    batch_size_ = new_batch_size;
    total_nb_frames_ = frame_capacity_ - (frame_capacity_ % batch_size_);

    CHECK(total_nb_frames_ > 0, "There must be more at least a frame in the queue.");

    max_size_ = total_nb_frames_ / batch_size_;

    batch_mutexes_ = std::unique_ptr<std::mutex[], std::default_delete<std::mutex[]>>(new std::mutex[max_size_]);
    if (device_ == Device::GPU)
    {
        batch_streams_ =
            std::unique_ptr<cudaStream_t[], std::default_delete<cudaStream_t[]>>(new cudaStream_t[max_size_]);

        for (uint i = 0; i < max_size_; ++i)
            cudaSafeCall(
                cudaStreamCreateWithPriority(&(batch_streams_[i]), cudaStreamDefault, CUDA_STREAM_QUEUE_PRIORITY));
    }

    data_.resize(static_cast<size_t>(max_size_) * batch_size_ * fd_.get_frame_size());

    const size_t meta_count = static_cast<size_t>(max_size_) * batch_size_;
    ids_.reset(new uint64_t[meta_count]);
    synced_ts_.reset(new uint64_t[meta_count]);
    camera_ts_.reset(new uint64_t[meta_count]);
    offset_ts_.reset(new uint64_t[meta_count]);
}

void BatchInputQueue::sync_current_batch() const
{
    if (curr_batch_counter_ > 0) // A batch is currently enqueued
        cudaXStreamSynchronize(batch_streams_[end_index_]);
    else if (end_index_ > 0) // No batch is enqueued, sync the last one
        cudaXStreamSynchronize(batch_streams_[end_index_ - 1]);
    else // if end_index_ == 0: sync the last index in queue (max_size_ - 1)
        cudaXStreamSynchronize(batch_streams_[max_size_ - 1]);
}

bool BatchInputQueue::is_current_batch_full() { return (curr_batch_counter_ == 0); }

void BatchInputQueue::destroy_mutexes_streams()
{
    if (device_ == Device::GPU)
    {
        for (uint i = 0; i < max_size_; i++)
            cudaSafeCall(cudaStreamSynchronize(batch_streams_[i]));
        for (uint i = 0; i < max_size_; i++)
            cudaSafeCall(cudaStreamDestroy(batch_streams_[i]));
        batch_streams_.reset(nullptr);
    }

    // All the mutexes are unlocked before deleting.
    batch_mutexes_.reset(nullptr);
}

void BatchInputQueue::reset_override()
{
    auto index = end_index_.load();

    // Use unique_lock to lock and automatically unlock the mutexes
    std::unique_lock<std::mutex> producer_lock(m_producer_busy_);
    std::unique_lock<std::mutex> batch_lock(batch_mutexes_[index]);

    // Reset the queue to its empty state
    make_empty();

    // Unlocks are handled by unique_lock going out of scope
}

void BatchInputQueue::make_empty()
{
    size_ = 0;
    curr_nb_frames_ = 0;
    start_index_ = 0;
    end_index_ = 0;
    curr_batch_counter_ = 0;
    has_overwritten_ = false;
}

void BatchInputQueue::stop_producer()
{
    if (curr_batch_counter_ != 0)
    {
        batch_mutexes_[end_index_].unlock();
        m_producer_busy_.unlock();
    }
}

void BatchInputQueue::enqueue(const void* const frames, const cudaMemcpyKind memcpy_kind, const int nb_frame)
{
    if ((memcpy_kind == cudaMemcpyDeviceToDevice || memcpy_kind == cudaMemcpyHostToDevice) && (device_ == Device::CPU))
        throw std::runtime_error("Input queue : can't cudaMemcpy to device with the queue on cpu");

    if ((memcpy_kind == cudaMemcpyDeviceToHost || memcpy_kind == cudaMemcpyHostToHost) && (device_ == Device::GPU))
        throw std::runtime_error("Input queue : can't cudaMemcpy to host with the queue on gpu");

    int frames_left = nb_frame;

    while (frames_left > 0)
    {
        if (curr_batch_counter_ == 0) // Enqueue in a new batch
        {
            // The producer might be descheduled before locking.
            // Consumer might modified curr_batch_counter_ here by the resize
            // However, it sets its value to 0 so the condition is still true.
            m_producer_busy_.lock();
            batch_mutexes_[end_index_].lock();
        }
        // Critical section between enqueue (producer) & resize (consumer)

        uint frames_to_enqueue = std::min(static_cast<uint>(frames_left), batch_size_ - curr_batch_counter_);

        // Static_cast to avoid overflow
        char* const new_frame_adress =
            data_.get() +
            ((static_cast<size_t>(end_index_) * batch_size_ + curr_batch_counter_) * fd_.get_frame_size());

        cudaXMemcpyAsync(new_frame_adress,
                         frames,
                         sizeof(char) * fd_.get_frame_size() * frames_to_enqueue,
                         memcpy_kind,
                         batch_streams_[end_index_]);

        // No sync needed here, the host doesn't need to wait for the copy to
        // end. Only the consumer needs to be sure the data is here before
        // manipulating it.

        // Increase the number of frames in the current batch
        curr_batch_counter_ += frames_to_enqueue;

        // The current batch is full
        if (curr_batch_counter_ == batch_size_)
        {
            curr_batch_counter_ = 0;
            const uint prev_end_index = end_index_;
            end_index_ = (end_index_ + 1) % max_size_;

            // The queue is full (in terms of batch)
            if (size_ == max_size_)
            {
                has_overwritten_ = true;
                start_index_ = (start_index_ + 1) % max_size_;
            }
            else
            {
                size_++;
                curr_nb_frames_ += batch_size_;
            }

            // Unlock the current batch mutex
            batch_mutexes_[prev_end_index].unlock();
            // No batch are busy anymore
            // End of critical section between enqueue (producer) & resize
            // (consumer)
            m_producer_busy_.unlock();

            // block the worker after completing a batch to allow the resizing of the queue.
            if (resize_in_progress_)
            {
                while (resize_in_progress_)
                    continue;
                return;
            }
        }
        frames_left -= frames_to_enqueue;
    }
}

void BatchInputQueue::dequeue(void* const dest, const camera::PixelDepth depth, const dequeue_func_t func)
{
    CHECK(size_ > 0);
    // Order cannot be guaranteed because of the try lock because a producer
    // might start enqueue between two try locks
    // Active waiting until the start batch is available to dequeue
    const uint start_index_locked = wait_and_lock(start_index_);

    // It is not needed to stream synchronize before the kernel call because it
    // is ran on the stream of the batch which will be blocking if operations
    // are still running.

    // From the queue
    const char* const src =
        data_.get() + (static_cast<size_t>(start_index_locked) * batch_size_ * fd_.get_frame_size());

    if (device_ == Device::GPU)
        func(src, dest, batch_size_, fd_.get_frame_res(), depth, batch_streams_[start_index_locked]);
    else
        func(src, dest, batch_size_, fd_.get_frame_res(), depth, 0);

    // The consumer has the responsability to give data that
    // finished processing.
    if (device_ == Device::GPU)
        cudaXStreamSynchronize(batch_streams_[start_index_locked]);

    // Update index
    dequeue_update_attr();

    // Unlock the dequeued batch
    batch_mutexes_[start_index_locked].unlock();
}

void BatchInputQueue::dequeue()
{
    // CHECK(size_ > 0);

    if (size_ > 0)
    {
        // Order cannot be guaranteed because of the try lock because a producer
        // might start enqueue between two try locks
        // Active waiting until the start batch is available to dequeue
        uint start_index_locked = wait_and_lock(start_index_);

        // Update index
        dequeue_update_attr();

        // Unlock the dequeued batch
        batch_mutexes_[start_index_locked].unlock();
    }
}

void BatchInputQueue::dequeue_update_attr()
{
    start_index_ = (start_index_ + 1) % max_size_;
    size_--;
    if (curr_nb_frames_ < batch_size_)
        curr_nb_frames_ = 0;
    else
        curr_nb_frames_ -= batch_size_;
}

void BatchInputQueue::rebuild(const camera::FrameDescriptor& fd,
                              const unsigned int size,
                              const unsigned int batch_size,
                              const Device device)
{
    set_fd(fd);
    if (device_ != device)
    {
        device_ = device;
        data_ = cuda_tools::UniquePtr<char>(device_);
    }

    frame_capacity_ = size;

    resize(batch_size);
}

void BatchInputQueue::resize(const uint new_batch_size)
{
    resize_in_progress_ = true;
    // No action on any batch must be proceed
    const std::lock_guard<std::mutex> lock(m_producer_busy_);
    // Critical section between the enqueue (producer) & the resize (consumer)
    // Synchronize all active CUDA streams
    if (device_ == Device::GPU)
    {
        for (uint i = 0; i < max_size_; ++i)
        {
            cudaSafeCall(cudaStreamSynchronize(batch_streams_[i]));
        }
    }
    // Every mutexes must be unlocked here.
    // Destroy all streams and mutexes that must be all unlocked
    destroy_mutexes_streams();

    // Create all streams and mutexes
    create_queue(new_batch_size);

    make_empty();
    // End of critical section
    resize_in_progress_ = false;
}

// void BatchInputQueue::dequeue(void* dest, const cudaStream_t stream, cudaMemcpyKind cuda_kind =
// cudaMemcpyDeviceToDevice)
// {

//     CHECK(size_ > 0, "Queue is empty. Cannot dequeue.");

//     // Order cannot be guaranteed because of the try lock because a producer
//     // might start enqueue between two try locks
//     // Active waiting until the start batch is available to dequeue
//     uint start_index_locked = wait_and_lock(start_index_);

//     Queue::MutexGuard m_guard_dst(dest.get_guard());

//     // Determine source region info
//     struct Queue::QueueRegion src;
//     // Get the start of the starting batch
//     src.first = data_.get() + (static_cast<size_t>(start_index_locked) * batch_size_ * fd_.get_frame_size());
//     // Copy multiple nb_elts which might be lower than batch_size.
//     src.first_size = nb_elts;

//     // Determine destination region info
//     struct Queue::QueueRegion dst;
//     const uint begin_to_enqueue_index = (dest.start_index_ + dest.size_) % dest.max_size_;

//     char* begin_to_enqueue = dest + (begin_to_enqueue_index * dest.fd_.get_frame_size());
//     if (begin_to_enqueue_index + nb_elts > dest.max_size_)
//     {
//         dst.first = begin_to_enqueue;
//         dst.first_size = dest.max_size_ - begin_to_enqueue_index;
//         dst.second = dest.data_.get();
//         dst.second_size = nb_elts - dst.first_size;
//     }
//     else
//     {
//         dst.first = begin_to_enqueue;
//         dst.first_size = nb_elts;
//     }

//     // Use the source start index (first batch of frames in the queue) stream
//     // An enqueue operation on this stream (if happens) is blocked until the
//     // copy is completed. Make the copy according to the region
//     if(device_ == Device::GPU)
//         Queue::copy_multiple_aux(src, dst, fd_.get_frame_size(), batch_streams_[start_index_locked], cuda_kind);
//     else
//         Queue::copy_multiple_aux(src, dst, fd_.get_frame_size(), 0, cuda_kind);

//     // The consumer has the responsability to give data that
//     // finished processing.
//     // would kill this queue design with only 1 producer and 1 consumer).
//     if(device_ == Device::GPU)
//         cudaXStreamSynchronize(batch_streams_[start_index_locked]);

//     // Update dest queue parameters
//     dest.size_ += nb_elts;

//     // Copy done, release the batch.
//     batch_mutexes_[start_index_locked].unlock();

//     if (dest.size_ > dest.max_size_)
//     {
//         dest.start_index_ = (dest.start_index_ + dest.size_) % dest.max_size_;
//         dest.size_.store(dest.max_size_.load());
//         dest.has_overwritten_ = true;
//     }

// }

void BatchInputQueue::copy_multiple(Queue& dest, cudaMemcpyKind cuda_kind)
{
    copy_multiple(dest, batch_size_, cuda_kind);
}

void BatchInputQueue::copy_multiple(Queue& dest, const uint nb_elts, cudaMemcpyKind cuda_kind)
{
    CHECK(size_ > 0, "Queue is empty. Cannot copy multiple.");
    CHECK(dest.get_max_size() >= nb_elts,
          "Copy multiple: the destination queue must have a size at least greater than number of elements to copy.");
    CHECK(fd_.get_frame_size() == dest.fd_.get_frame_size());
    CHECK(nb_elts <= batch_size_, "Copy multiple: cannot copy more than a batch of frames");

    // Order cannot be guaranteed because of the try lock because a producer
    // might start enqueue between two try locks
    // Active waiting until the start batch is available to dequeue
    uint start_index_locked = wait_and_lock(start_index_);

    Queue::MutexGuard m_guard_dst(dest.get_guard());

    // Determine source region info
    struct Queue::QueueRegion src;
    // Get the start of the starting batch
    src.first = data_.get() + (static_cast<size_t>(start_index_locked) * batch_size_ * fd_.get_frame_size());
    // Copy multiple nb_elts which might be lower than batch_size.
    src.first_size = nb_elts;

    if (API.record.get_frame_acquisition_enabled() && API.record.get_record_mode() == RecordMode::RAW)
    {
        const size_t base = static_cast<size_t>(start_index_locked) * batch_size_;
        g_record_stamp_queue.push_range_from_arrays(ids_.get() + base,
                                                    synced_ts_.get() + base,
                                                    camera_ts_.get() + base,
                                                    offset_ts_.get() + base,
                                                    nb_elts);
    }

    // Determine destination region info
    struct Queue::QueueRegion dst;
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

    // Use the source start index (first batch of frames in the queue) stream
    // An enqueue operation on this stream (if happens) is blocked until the
    // copy is completed. Make the copy according to the region
    if (device_ == Device::GPU)
        Queue::copy_multiple_aux(src, dst, fd_.get_frame_size(), batch_streams_[start_index_locked], cuda_kind);
    else
        Queue::copy_multiple_aux(src, dst, fd_.get_frame_size(), 0, cuda_kind);

    // As in dequeue, the consumer has the responsability to give data that
    // finished processing.
    // would kill this queue design with only 1 producer and 1 consumer).
    if (device_ == Device::GPU)
        cudaXStreamSynchronize(batch_streams_[start_index_locked]);

    // Update dest queue parameters
    dest.size_ += nb_elts;

    // Copy done, release the batch.
    batch_mutexes_[start_index_locked].unlock();

    if (dest.size_ >= dest.max_size_)
    {
        dest.start_index_ = (dest.start_index_ + dest.size_) % dest.max_size_;
        dest.size_.store(dest.max_size_.load());
        dest.has_overwritten_ = true;
    }
}

void BatchInputQueue::enqueue_with_ids(const void* const frames,
                                       const cudaMemcpyKind memcpy_kind,
                                       const int nb_frame,
                                       uint64_t base_id,
                                       uint64_t ts0_synced_us,
                                       uint64_t period_us,
                                       uint64_t ts0_cam_us,
                                       uint64_t offset_us)
{
    // Validate copy direction vs. queue device
    if ((memcpy_kind == cudaMemcpyDeviceToDevice || memcpy_kind == cudaMemcpyHostToDevice) && (device_ == Device::CPU))
        throw std::runtime_error("Input queue: cannot cudaMemcpy to device when the queue is on CPU");

    if ((memcpy_kind == cudaMemcpyDeviceToHost || memcpy_kind == cudaMemcpyHostToHost) && (device_ == Device::GPU))
        throw std::runtime_error("Input queue: cannot cudaMemcpy to host when the queue is on GPU");

    // Source base pointer and sizes
    const size_t bytes_per_frame = static_cast<size_t>(fd_.get_frame_size());
    const char* const src_frames_base = static_cast<const char*>(frames);

    int frames_left = nb_frame;
    size_t frames_copied = 0; // how many source frames have already been copied

    // IDs/timestamps to write for the next frame
    uint64_t next_id = base_id;
    uint64_t next_synced = ts0_synced_us;
    uint64_t next_cam = ts0_cam_us;

    while (frames_left > 0)
    {
        // Lock once when starting a new batch
        if (curr_batch_counter_ == 0)
        {
            m_producer_busy_.lock();
            batch_mutexes_[end_index_].lock();
        }

        // Number of frames to enqueue in this iteration
        const uint frames_to_enqueue = std::min(static_cast<uint>(frames_left), batch_size_ - curr_batch_counter_);

        // Destination pointer for this slice
        char* const dst_frames =
            data_.get() + ((static_cast<size_t>(end_index_) * batch_size_ + curr_batch_counter_) * bytes_per_frame);

        // advance the source pointer according to already-copied frames
        const char* const src_frames = src_frames_base + frames_copied * bytes_per_frame;

        // Async copy of image bytes
        cudaXMemcpyAsync(dst_frames,
                         src_frames,
                         bytes_per_frame * static_cast<size_t>(frames_to_enqueue),
                         memcpy_kind,
                         batch_streams_[end_index_]);

        // Write metadata (host)
        const size_t meta_base = static_cast<size_t>(end_index_) * batch_size_ + curr_batch_counter_;
        uint64_t* const dst_ids = ids_.get() + meta_base;
        uint64_t* const dst_synced = synced_ts_.get() + meta_base;
        uint64_t* const dst_cam = camera_ts_.get() + meta_base;
        uint64_t* const dst_off = offset_ts_.get() + meta_base;

        for (uint i = 0; i < frames_to_enqueue; ++i)
        {
            dst_ids[i] = next_id + i;
            dst_synced[i] = next_synced + static_cast<uint64_t>(i) * period_us;
            dst_cam[i] = ts0_cam_us ? (next_cam + static_cast<uint64_t>(i) * period_us) : 0;
            dst_off[i] = offset_us;
        }

        // Advance ID/time cursors for the next slice
        next_id += frames_to_enqueue;
        next_synced += static_cast<uint64_t>(frames_to_enqueue) * period_us;
        if (ts0_cam_us)
            next_cam += static_cast<uint64_t>(frames_to_enqueue) * period_us;

        // Advance source-frame counters and remaining frames
        frames_copied += static_cast<size_t>(frames_to_enqueue);
        frames_left -= static_cast<int>(frames_to_enqueue);

        // Batch bookkeeping
        curr_batch_counter_ += frames_to_enqueue;

        if (curr_batch_counter_ == batch_size_)
        {
            curr_batch_counter_ = 0;
            const uint prev_end_index = end_index_;
            end_index_ = (end_index_ + 1) % max_size_;

            if (size_ == max_size_)
            {
                has_overwritten_ = true;
                start_index_ = (start_index_ + 1) % max_size_;
            }
            else
            {
                size_++;
                curr_nb_frames_ += batch_size_;
            }

            // copies are async; the consumer must synchronize the corresponding stream
            batch_mutexes_[prev_end_index].unlock();
            m_producer_busy_.unlock();

            if (resize_in_progress_)
            {
                // Busy-wait until the resize finishes, then exit early
                while (resize_in_progress_)
                    continue;
                return;
            }
        }
    }
}

} // namespace holovibes
