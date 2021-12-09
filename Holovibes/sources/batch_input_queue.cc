#include <cassert>

#include "holovibes.hh"

#include "batch_input_queue.hh"

namespace holovibes
{
BatchInputQueue::BatchInputQueue(const uint total_nb_frames, const uint batch_size, const camera::FrameDescriptor& fd)
    : DisplayQueue(fd)
    , fast_updates_entry_(GSH::fast_updates_map<QueueType>.create_entry(QueueType::INPUT_QUEUE))
    , curr_nb_frames_(fast_updates_entry_->first)
    , total_nb_frames_(fast_updates_entry_->second)
    , frame_capacity_(total_nb_frames)
{
    curr_nb_frames_ = 0;
    total_nb_frames_ = total_nb_frames;

    // Set priority of streams
    // Set batch_size and max_size
    create_queue(batch_size);
}

BatchInputQueue::~BatchInputQueue()
{
    destroy_mutexes_streams();
    // data is free as it is a UniquePtr.

    GSH::fast_updates_map<QueueType>.remove_entry(QueueType::INPUT_QUEUE);
}

void BatchInputQueue::create_queue(const uint new_batch_size)
{
    CHECK(new_batch_size > 0) << "Batch size cannot be 0.";

    batch_size_ = new_batch_size;
    total_nb_frames_ = frame_capacity_ - (frame_capacity_ % batch_size_);

    CHECK(total_nb_frames_ > 0) << "There must be more at least a frame in the queue.";

    max_size_ = total_nb_frames_ / batch_size_;

    batch_mutexes_ = std::unique_ptr<std::mutex[]>(new std::mutex[max_size_]);
    batch_streams_ = std::unique_ptr<cudaStream_t[]>(new cudaStream_t[max_size_]);

    for (uint i = 0; i < max_size_; ++i)
        cudaSafeCall(cudaStreamCreateWithPriority(&(batch_streams_[i]), cudaStreamDefault, CUDA_STREAM_QUEUE_PRIORITY));

    data_.resize(static_cast<size_t>(max_size_) * batch_size_ * fd_.get_frame_size());
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
    for (uint i = 0; i < max_size_; i++)
        cudaSafeCall(cudaStreamSynchronize(batch_streams_[i]));
    for (uint i = 0; i < max_size_; i++)
        cudaSafeCall(cudaStreamDestroy(batch_streams_[i]));
    batch_streams_.reset(nullptr);

    // All the mutexes are unlocked before deleting.
    batch_mutexes_.reset(nullptr);
}

void BatchInputQueue::make_empty()
{
    size_ = 0;
    curr_nb_frames_ = 0;
    start_index_ = 0;
    end_index_ = 0;
    curr_batch_counter_ = 0;
    has_overridden_ = false;
}

void BatchInputQueue::stop_producer()
{
    if (curr_batch_counter_ != 0)
    {
        batch_mutexes_[end_index_].unlock();
        m_producer_busy_.unlock();
    }
}

void BatchInputQueue::enqueue(const void* const input_frame, const cudaMemcpyKind memcpy_kind)
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

    // Static_cast to avoid overflow
    char* const new_frame_adress =
        data_.get() + ((static_cast<size_t>(end_index_) * batch_size_ + curr_batch_counter_) * fd_.get_frame_size());

    LOG_DEBUG << "start_index : " << start_index_ << "\nend_index : " << end_index_;

    cudaXMemcpyAsync(new_frame_adress,
                     input_frame,
                     sizeof(char) * fd_.get_frame_size(),
                     memcpy_kind,
                     batch_streams_[end_index_]);

    // No sync needed here, the host doesn't need to wait for the copy to
    // end. Only the consumer needs to be sure the data is here before
    // manipulating it.

    // Increase the number of frames in the current batch
    curr_batch_counter_++;

    // The current batch is full
    if (curr_batch_counter_ == batch_size_)
    {
        curr_batch_counter_ = 0;
        const uint prev_end_index = end_index_;
        end_index_ = (end_index_ + 1) % max_size_;
        LOG_DEBUG << "new end index :" << end_index_;

        // The queue is full (in terms of batch)
        if (size_ == max_size_)
        {
            has_overridden_ = true;
            start_index_ = (start_index_ + 1) % max_size_;
            LOG_DEBUG << "new start index :" << start_index_;
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
    }
}

void BatchInputQueue::dequeue(void* const dest, const uint depth, const dequeue_func_t func)
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
    func(src, dest, batch_size_, fd_.get_frame_res(), depth, batch_streams_[start_index_locked]);

    // The consumer has the responsability to give data that
    // finished processing.
    cudaXStreamSynchronize(batch_streams_[start_index_locked]);

    // Update index
    dequeue_update_attr();

    // Unlock the dequeued batch
    batch_mutexes_[start_index_locked].unlock();
}

void BatchInputQueue::dequeue()
{
    CHECK(size_ > 0);
    // Order cannot be guaranteed because of the try lock because a producer
    // might start enqueue between two try locks
    // Active waiting until the start batch is available to dequeue
    uint start_index_locked = wait_and_lock(start_index_);

    // Update index
    dequeue_update_attr();

    // Unlock the dequeued batch
    batch_mutexes_[start_index_locked].unlock();
}

void BatchInputQueue::dequeue_update_attr()
{
    start_index_ = (start_index_ + 1) % max_size_;
    size_--;
    curr_nb_frames_ -= batch_size_;
}

void BatchInputQueue::resize(const uint new_batch_size)
{
    // No action on any batch must be proceed
    const std::lock_guard<std::mutex> lock(m_producer_busy_);

    // Critical section between the enqueue (producer) & the resize (consumer)

    // Every mutexes must be unlocked here.

    // Destroy all streams and mutexes that must be all unlocked
    destroy_mutexes_streams();

    // Create all streams and mutexes
    create_queue(new_batch_size);

    make_empty();

    // End of critical section
}

void BatchInputQueue::copy_multiple(Queue& dest) { copy_multiple(dest, batch_size_); }

void BatchInputQueue::copy_multiple(Queue& dest, const uint nb_elts)
{
    CHECK(size_ > 0) << "Queue is empty. Cannot copy multiple.";
    CHECK(dest.get_max_size() >= nb_elts) << "Copy multiple: the destination queue must have a size at least "
                                             "greater than number of elements to copy.";
    CHECK(fd_.get_frame_size() == dest.fd_.get_frame_size());
    CHECK(nb_elts <= batch_size_) << "Copy multiple: cannot copy more "
                                     "than a batch of frames";

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
    Queue::copy_multiple_aux(src, dst, fd_.get_frame_size(), batch_streams_[start_index_locked]);

    // As in dequeue, the consumer has the responsability to give data that
    // finished processing.
    // (Stream synchronization could only be done in thread recorder but it
    // would kill this queue design with only 1 producer and 1 consumer).
    cudaXStreamSynchronize(batch_streams_[start_index_locked]);

    // Update dest queue parameters
    dest.size_ += nb_elts;

    // Copy done, release the batch.
    batch_mutexes_[start_index_locked].unlock();

    if (dest.size_ > dest.max_size_)
    {
        dest.start_index_ = (dest.start_index_ + dest.size_) % dest.max_size_;
        dest.size_.store(dest.max_size_.load());
        dest.has_overridden_ = true;
    }
}
} // namespace holovibes
