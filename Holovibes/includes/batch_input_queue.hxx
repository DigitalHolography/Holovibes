#pragma once

#include <cassert>

#include "cuda_memory.cuh"
#include "batch_input_queue.hh"

namespace holovibes
{

template <typename T>
BatchInputQueue<T>::BatchInputQueue(const uint total_nb_frames, const uint batch_size, const uint frame_res)
    : frame_res_(frame_res)
{
    // Set priority of streams
    create_mutexes_streams(total_nb_frames, batch_size);
    cudaXMalloc(&data_, static_cast<size_t>(max_size_) * batch_size_ * frame_res_ * sizeof(T));
}

template <typename T>
BatchInputQueue<T>::~BatchInputQueue()
{
    destroy_mutexes_streams();
    cudaXFree(data_);
}

template <typename T>
void BatchInputQueue<T>::create_mutexes_streams(const uint total_nb_frames, const uint new_batch_size)
{
    assert(new_batch_size > 0 && "Batch size cannot be 0.");
    assert(total_nb_frames > 0 && "There must be more at least a frame in the queue.");
    assert(total_nb_frames % new_batch_size == 0 && "Queue size must be a submultiple of batch size.");

    batch_size_ = new_batch_size;
    max_size_ = total_nb_frames / batch_size_;

    batch_mutexes_ = std::unique_ptr<std::mutex[]>(new std::mutex[max_size_]);
    batch_streams_ = std::unique_ptr<cudaStream_t[]>(new cudaStream_t[max_size_]);
    for (uint i = 0; i < max_size_; ++i)
        cudaSafeCall(cudaStreamCreate(&(batch_streams_[i])));
}

template <typename T>
void BatchInputQueue<T>::destroy_mutexes_streams()
{
    for (uint i = 0; i < max_size_; i++)
        cudaSafeCall(cudaStreamSynchronize(batch_streams_[i]));
    for (uint i = 0; i < max_size_; i++)
        cudaSafeCall(cudaStreamDestroy(batch_streams_[i]));
    batch_streams_.reset(nullptr);

    // All the mutexes are unlocked before deleting.
    batch_mutexes_.reset(nullptr);
}

template <typename T>
void BatchInputQueue<T>::make_empty()
{
    size_ = 0;
    start_index_ = 0;
    end_index_ = 0;
    curr_batch_counter_ = 0;
    has_overridden_ = false;
}

template <typename T>
void BatchInputQueue<T>::stop_producer()
{
    if (curr_batch_counter_ != 0)
    {
        batch_mutexes_[end_index_].unlock();
        m_producer_busy_.unlock();
    }
}

template <typename T>
void BatchInputQueue<T>::enqueue(const T* const input_frame,
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
    T* const new_frame_adress =
        data_ + ((static_cast<size_t>(end_index_) * batch_size_ + curr_batch_counter_) * frame_res_);

    cudaMemcpyXAsync(new_frame_adress,
                     input_frame,
                     sizeof(T) * frame_res_,
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
        // End of critical section between enqueue (producer) & resize (consumer)
        m_producer_busy_.unlock();
    }
}


template <typename T>
template <typename FUNC>
void BatchInputQueue<T>::dequeue(T* const dest, FUNC func)
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
    const T* const src = data_ + (static_cast<size_t>(start_index_) * batch_size_ * frame_res_);
    func(src, dest, batch_size_, frame_res_, batch_streams_[start_index_]);

    // The consumer has the responsability to give data that
    // finished processing.
    cudaXStreamSynchronize(batch_streams_[start_index_]);

    // Update index
    const uint prev_start_index = start_index_;
    start_index_ = (start_index_ + 1) % max_size_;
    size_--;

    // Unlock the dequeued batch
    batch_mutexes_[prev_start_index].unlock();
}

template <typename T>
void BatchInputQueue<T>::resize(const uint new_batch_size)
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

template <typename T>
void BatchInputQueue<T>::copy_multiple(BatchInputQueue& dest)
{
    // HOLO: Make sure the sizes match
    // assert(frame_res_ * sizeof(T) == dest.frame_size_);
    assert(size_ > 0 && "Queue is empty. Cannot copy multiple.");
    assert(dest.get_max_size() >= batch_size_
        && "Copy multiple: the destination queue must have a size at least greater than batch_size");

    // Order cannot be guaranteed because of the try lock because a producer
    // might start enqueue between two try locks
    // Active waiting until the start batch is available to dequeue
    while (!batch_mutexes_[start_index_].try_lock())
        continue;

    // HOLO: Write the following lock to lock the regular dst queue
    // MutexGuard m_guard_dst(dest.get_guard());

    // Determine source region info
    // Can't use QueueRegion<T> in order to be consistent with the other
    // Queue using char*. It will avoid some frame_res/frame_size issue
    struct QueueRegion<char> src;
    char* const copy_start = reinterpret_cast<char*>(
        data_ + (static_cast<size_t>(start_index_) * frame_res_));
    src.first = copy_start;
    src.first_size = nb_elts;

    // Determine destination region info
    // HOLO: Use getters
    struct QueueRegion<char> dst;
    const uint begin_to_enqueue_index =
        (dest.start_index_ + dest.size_) % dest.max_size_;

    // HOLO: replace frame_res by frame_size. Double check the types
    // conversion and everything HOLO: Data should directly be char* so
    // remove static_cast
    char* begin_to_enqueue =
        dest.data_ + (begin_to_enqueue_index * dest.frame_res_);
    if (begin_to_enqueue_index + batch_size_ > dest.max_size_)
    {
        dst.first = begin_to_enqueue;
        dst.first_size = dest.max_size_ - begin_to_enqueue_index;
        dst.second = dest.data_;
        dst.second_size = batch_size_ - dst.first_size;
    }
    else
    {
        dst.first = begin_to_enqueue;
        dst.first_size = batch_size_;
    }

    // Use the source start index (first batch of frames in the queue) stream
    // An enqueue operation on this stream (if happens) is blocked until the copy is completed.
    // Make the copy according to the region
    copy_multiple_aux(src, dst, batch_streams_[start_index_]);

    // As in dequeue, the consumer has the responsability to give data that
    // finished processing.
    // (Stream synchronization could only be done in thread recorder but it
    // would kill this queue design with only 1 producer and 1 consumer).
    cudaXStreamSynchronize(batch_streams_[start_index_]);

    // Update dest queue parameters
    dest.size_ += batch_size_;
    if (dest.size_ > dest.max_size_)
    {
        dest.start_index_ = (dest.start_index_ + dest.size_) % dest.max_size_;
        // HOLO: dest.size_.store(dest.max_size_.load());
        dest.size_.store(dest.max_size_);
        dest.has_overridden_ = true;
    }
}

template <typename T>
template <typename U>
void BatchInputQueue<T>::copy_multiple_aux(QueueRegion<U>& src,
                                           QueueRegion<U>& dst,
                                           cudaStream_t copying_stream)
{
    // Use frame_size to comply with the regular queue that unfortunately
    // uses char*
    const size_t frame_size = static_cast<size_t>(frame_res_) * sizeof(T);

    // Handle copies depending on regions info
    if (src.overflow())
    {
        if (dst.overflow())
        {
            if (src.first_size > dst.first_size)
            {
                cudaXMemcpyAsync(dst.first,
                                 src.first,
                                 dst.first_size * frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 copying_stream);
                src.consume_first(dst.first_size, frame_size);

                cudaXMemcpyAsync(dst.second,
                                 src.first,
                                 src.first_size * frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 copying_stream);
                dst.consume_second(src.first_size, frame_size);

                cudaXMemcpyAsync(dst.second,
                                 src.second,
                                 src.second_size * frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 copying_stream);
            }
            else // src.first_size <= dst.first_size
            {
                cudaXMemcpyAsync(dst.first,
                                 src.first,
                                 src.first_size * frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 copying_stream);
                dst.consume_first(src.first_size, frame_size);

                if (src.second_size > dst.first_size)
                {
                    cudaXMemcpyAsync(dst.first,
                                     src.second,
                                     dst.first_size * frame_size,
                                     cudaMemcpyDeviceToDevice,
                                     copying_stream);
                    src.consume_second(dst.first_size, frame_size);

                    cudaXMemcpyAsync(dst.second,
                                     src.second,
                                     src.second_size * frame_size,
                                     cudaMemcpyDeviceToDevice,
                                     copying_stream);
                }
                else // src.second_size == dst.first_size
                {
                    cudaXMemcpyAsync(dst.first,
                                     src.second,
                                     src.second_size * frame_size,
                                     cudaMemcpyDeviceToDevice,
                                     copying_stream);
                }
            }
        }
        else
        {
            // In this case: dst.first_size > src.first_size

            cudaXMemcpyAsync(dst.first,
                             src.first,
                             src.first_size * frame_size,
                             cudaMemcpyDeviceToDevice,
                             copying_stream);
            dst.consume_first(src.first_size, frame_size);

            cudaXMemcpyAsync(dst.first,
                             src.second,
                             dst.first_size * frame_size,
                             cudaMemcpyDeviceToDevice,
                             copying_stream);
        }
    }
    else
    {
        if (dst.overflow())
        {
            // In this case: src.first_size > dst.first_size

            cudaXMemcpyAsync(dst.first,
                             src.first,
                             dst.first_size * frame_size,
                             cudaMemcpyDeviceToDevice,
                             copying_stream);
            src.consume_first(dst.first_size, frame_size);

            cudaXMemcpyAsync(dst.second,
                             src.first,
                             src.first_size * frame_size,
                             cudaMemcpyDeviceToDevice,
                             copying_stream);
        }
        else
        {
            cudaXMemcpyAsync(dst.first,
                             src.first,
                             src.first_size * frame_size,
                             cudaMemcpyDeviceToDevice,
                             copying_stream);
        }
    }
}

template <typename T>
inline bool BatchInputQueue<T>::is_empty() const
{
    return size_ == 0;
}

template <typename T>
inline uint BatchInputQueue<T>::get_size() const
{
    return size_;
}

template <typename T>
inline bool BatchInputQueue<T>::has_overridden() const
{
    return has_overridden_;
}

template <typename T>
inline const T* BatchInputQueue<T>::get_data() const
{
    return data_;
}

template <typename T>
inline uint BatchInputQueue<T>::get_frame_res() const
{
    return frame_res_;
}
} // namespace holovibes