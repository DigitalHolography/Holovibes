/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include <cassert>

#include "holovibes.hh"

#include "cuda_memory.cuh"
#include "batch_input_queue.hh"

namespace holovibes
{
BatchInputQueue::BatchInputQueue(const uint total_nb_frames,
                                 const camera::FrameDescriptor& fd)
    : BatchInputQueue(total_nb_frames, 1, fd)
{
}

BatchInputQueue::BatchInputQueue(const uint total_nb_frames,
                                 const uint batch_size,
                                 const camera::FrameDescriptor& fd)
    : fd_(fd)
    , frame_res_(fd_.frame_res())
    , frame_size_(fd_.frame_size())
    , total_nb_frames_(total_nb_frames)
{
    // Set priority of streams
    // Set batch_size and max_size
    create_mutexes_streams(total_nb_frames, batch_size);
    cudaXMalloc(&data_,
                static_cast<size_t>(max_size_) * batch_size_ * frame_size_);

    Holovibes::instance().get_info_container().add_queue_size(
        Queue::QueueType::INPUT_QUEUE,
        curr_nb_frames_,
        total_nb_frames_);
}

BatchInputQueue::~BatchInputQueue()
{
    destroy_mutexes_streams();
    cudaXFree(data_);

    Holovibes::instance().get_info_container().remove_queue_size(
        Queue::QueueType::INPUT_QUEUE);
}

void BatchInputQueue::create_mutexes_streams(const uint total_nb_frames,
                                             const uint new_batch_size)
{
    assert(new_batch_size > 0 && "Batch size cannot be 0.");
    assert(total_nb_frames > 0 &&
           "There must be more at least a frame in the queue.");
    assert(total_nb_frames % new_batch_size == 0 &&
           "Queue size must be a submultiple of batch size.");

    batch_size_ = new_batch_size;
    max_size_ = total_nb_frames / batch_size_;

    batch_mutexes_ = std::unique_ptr<std::mutex[]>(new std::mutex[max_size_]);
    batch_streams_ =
        std::unique_ptr<cudaStream_t[]>(new cudaStream_t[max_size_]);
    for (uint i = 0; i < max_size_; ++i)
        cudaSafeCall(cudaStreamCreate(&(batch_streams_[i])));
}

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

void BatchInputQueue::enqueue(const void* const input_frame,
                              const cudaMemcpyKind memcpy_kind)
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
        data_ +
        ((static_cast<size_t>(end_index_) * batch_size_ + curr_batch_counter_) *
         frame_size_);

    cudaXMemcpyAsync(new_frame_adress,
                     input_frame,
                     sizeof(char) * frame_size_,
                     memcpy_kind,
                     batch_streams_[end_index_]);

    // No sync needed here, the host doesn't need to wait for the copy to
    // end. Only the consumer needs to be sure the data is here before
    // manipulating it.

    // Increase the number of frames in the current batch
    curr_batch_counter_++;
    curr_nb_frames_++;

    // The current batch is full
    if (curr_batch_counter_ == batch_size_)
    {
        curr_batch_counter_ = 0;
        const uint prev_end_index = end_index_;
        end_index_ = (end_index_ + 1) % max_size_;

        // The queue is full (in terms of batch)
        if (size_ == max_size_)
        {
            has_overridden_ = true;
            start_index_ = (start_index_ + 1) % max_size_;
        }
        else
            size_++;

        // Unlock the current batch mutex
        batch_mutexes_[prev_end_index].unlock();
        // No batch are busy anymore
        // End of critical section between enqueue (producer) & resize
        // (consumer)
        m_producer_busy_.unlock();
    }
}

void BatchInputQueue::dequeue(void* const dest,
                              const uint depth,
                              const dequeue_func_t func)
{
    assert(size_ > 0);
    // Order cannot be guaranteed because of the try lock because a producer
    // might start enqueue between two try locks
    // Active waiting until the start batch is available to dequeue
    while (!batch_mutexes_[start_index_].try_lock())
        continue;

    // It is not needed to stream synchronize before the kernel call because it
    // is ran on the stream of the batch which will be blocking if operations
    // are still running.

    // From the queue
    const char* const src =
        data_ + (static_cast<size_t>(start_index_) * batch_size_ * frame_size_);
    func(src,
         dest,
         batch_size_,
         frame_res_,
         depth,
         batch_streams_[start_index_]);

    // The consumer has the responsability to give data that
    // finished processing.
    cudaXStreamSynchronize(batch_streams_[start_index_]);

    // Update index
    const uint prev_start_index = start_index_;
    dequeue_update_attr();

    // Unlock the dequeued batch
    batch_mutexes_[prev_start_index].unlock();
}

void BatchInputQueue::dequeue()
{
    assert(size_ > 0);
    // Order cannot be guaranteed because of the try lock because a producer
    // might start enqueue between two try locks
    // Active waiting until the start batch is available to dequeue
    while (!batch_mutexes_[start_index_].try_lock())
        continue;

    // Update index
    const uint prev_start_index = start_index_;
    dequeue_update_attr();

    // Unlock the dequeued batch
    batch_mutexes_[prev_start_index].unlock();
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
    const uint total_nb_frames = batch_size_ * max_size_;
    create_mutexes_streams(total_nb_frames, new_batch_size);

    make_empty();

    // End of critical section
}

void BatchInputQueue::copy_multiple(Queue& dest)
{
    assert(size_ > 0 && "Queue is empty. Cannot copy multiple.");
    assert(dest.get_max_size() >= batch_size_ &&
           "Copy multiple: the destination queue must have a size at least "
           "greater than batch_size.");
    assert(frame_size_ == dest.frame_size_);

    // Order cannot be guaranteed because of the try lock because a producer
    // might start enqueue between two try locks
    // Active waiting until the start batch is available to dequeue
    while (!batch_mutexes_[start_index_].try_lock())
        continue;

    Queue::MutexGuard m_guard_dst(dest.get_guard());

    // Determine source region info
    struct Queue::QueueRegion src;
    src.first =
        data_ + (static_cast<size_t>(start_index_) * batch_size_ * frame_size_);
    src.first_size = batch_size_;

    // Determine destination region info
    struct Queue::QueueRegion dst;
    const uint begin_to_enqueue_index =
        (dest.start_index_ + dest.size_) % dest.max_size_;

    char* begin_to_enqueue =
        dest.data_.get() + (begin_to_enqueue_index * dest.frame_size_);
    if (begin_to_enqueue_index + batch_size_ > dest.max_size_)
    {
        dst.first = begin_to_enqueue;
        dst.first_size = dest.max_size_ - begin_to_enqueue_index;
        dst.second = dest.data_.get();
        dst.second_size = batch_size_ - dst.first_size;
    }
    else
    {
        dst.first = begin_to_enqueue;
        dst.first_size = batch_size_;
    }

    // Use the source start index (first batch of frames in the queue) stream
    // An enqueue operation on this stream (if happens) is blocked until the
    // copy is completed. Make the copy according to the region
    Queue::copy_multiple_aux(src,
                             dst,
                             frame_size_,
                             batch_streams_[start_index_]);

    // As in dequeue, the consumer has the responsability to give data that
    // finished processing.
    // (Stream synchronization could only be done in thread recorder but it
    // would kill this queue design with only 1 producer and 1 consumer).
    cudaXStreamSynchronize(batch_streams_[start_index_]);

    // Update dest queue parameters
    dest.size_ += batch_size_;

    // Copy done, release the batch.
    batch_mutexes_[start_index_].unlock();

    if (dest.size_ > dest.max_size_)
    {
        dest.start_index_ = (dest.start_index_ + dest.size_) % dest.max_size_;
        dest.size_.store(dest.max_size_.load());
        dest.has_overridden_ = true;
    }
}
} // namespace holovibes