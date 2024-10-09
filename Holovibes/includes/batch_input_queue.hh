/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <atomic>
#include <mutex>

#include "cuda_memory.cuh"
#include "display_queue.hh"
#include "queue.hh"
#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "global_state_holder.hh"
#include "enum_device.hh"

using uint = unsigned int;

namespace holovibes
{

class Queue;

/*! \class BatchInputQueue
 *
 * \brief Circular queue to handle CPU and GPU data, split into thread-safe batches, so that different batches can be
 * read and written simultaneously
 *
 * Conditons:
 *   2 threads: 1 Consumer (dequeue, copy multiple) and 1 producer
 *   batch size in the queue must be a submultiple of the queue size
 *   The order of operations cannot be guaranteed if the queue is full
 *   i.g. Enqueue, Dequeue, Enqueue might be processed in this order Enqueue,
 *   Dequeue, Enqueue
 */
class BatchInputQueue final : public DisplayQueue
{
  public: /* Public methods */
    BatchInputQueue(const uint total_nb_frames,
                    const uint batch_size,
                    const camera::FrameDescriptor& fd,
                    const Device device = Device::GPU);

    ~BatchInputQueue();

    /*! \brief Enqueue a frame in the queue
     *
     * Called by the producer.
     * The producer is in the critical while enqueueing in a batch
     * and exit this critical section when a batch of frames is full
     * in order to let the resize occure if needed.
     */
    void enqueue(const void* const input_frame, const cudaMemcpyKind memcpy_kind = cudaMemcpyDeviceToDevice);

    // bool enqueue(void* elt, const cudaStream_t stream, const cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    /*! \brief Copy multiple
     *
     * Called by the consumer.
     *
     * \param dest The destination queue
     * \param nb_elts Number of elts to copy multiple (must be lower than batch_size)
     */
    void copy_multiple(Queue& dest, const uint nb_elts, cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    /*! \brief Copy multiple
     *
     * Called by the consumer.
     * Call copy multiple with nb_elts == batch_size_
     *
     * \param dest The destination queue
     */
    void copy_multiple(Queue& dest, cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    /*! \brief Function used when dequeuing a batch of frame
     *
     * src, dst, batch_size, frame_res, depth, stream -> void
     */
    using dequeue_func_t = std::function<void(
        const void* const, void* const, const uint, const size_t, const camera::PixelDepth, const cudaStream_t)>;

    /*! \brief Deqeue a batch of frames. Block until the queue has at least a full batch of frame.
     *
     * The queue must have at least a batch of frames filled
     * Called by the consumer.
     *
     * \param dest Dequeue in the destination buffer
     * \param depth Depth of frame
     * \param func Apply a function to the batch of frames being dequeued
     */
    void dequeue(void* const dest, const camera::PixelDepth depth, const dequeue_func_t func);

    // void dequeue(void* dest, const cudaStream_t stream, cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice)
    // override;

    /*! \brief Deqeue a batch of frames. Block until the queue has at least a full batch of frame.
     *
     * The queue must have at least a batch of frames filled
     * Called by the consumer
     */
    void dequeue();

    /*!
     * \brief Rebuild the queue (change the fd or the device on which it is allocated), without creating a new queue.
     * Useful to keep using the pointer.
     *
     * \param fd
     * \param size
     * \param gpu
     */
    void rebuild(const camera::FrameDescriptor& fd,
                 const unsigned int size,
                 const unsigned int batch_size,
                 const Device device);

    /*! \brief Resize with a new batch size
     *
     * Called by the consumer.
     * Empty the queue.
     */
    void resize(const uint new_batch_size);

    /*! \brief Stop the producer.
     *
     * Exit the critical section of the producer.
     * It is required because the producer can only exit the critical section only when a batch is full.
     * However, the producer will stop its enqueues (in Holovibes the reader is stopped)
     * and will not be able to exit the critical section in case the current batch is still not full.
     * The queue needs this critical section to be exited for its destruction later.
     */
    void stop_producer();

    // void start_producer();

    void sync_current_batch() const;

    bool is_current_batch_full();

    inline void* get_last_image() const override
    {
        if (device_ == Device::GPU)
        {
            const std::lock_guard<std::mutex> lock(m_producer_busy_);
            sync_current_batch();
        }
        // Return the previous enqueued frame
        return data_.get() + ((start_index_ + curr_nb_frames_ - 1) % total_nb_frames_) * fd_.get_frame_size();
    }

    bool is_empty() const { return size_ == 0; }

    uint get_size() const { return size_; }

    uint get_max_size() const { return max_size_; }

    bool has_overwritten() const { return has_overwritten_; }

    void reset_override();

    // HOLO: Can it be removed?
    const void* get_data() const { return data_; }

    uint get_total_nb_frames() const { return total_nb_frames_; }

    const camera::FrameDescriptor& get_fd() const { return fd_; }

  private: /* Private methods */
    /*! \brief Set size attributes and create mutexes and streams arrays.
     *
     * Used by the consumer and constructor
     *
     * \param new_batch_size The new number of frames in a batch
     */
    void create_queue(const uint new_batch_size);

    /*! \brief Destroy mutexes and streams arrays.
     *
     * Used by the consumer and constructor.
     */
    void destroy_mutexes_streams();

    /*! \brief Update queue attributes to make the queue empty
     *
     * Used by the consumer.
     */
    void make_empty();

    /*! \brief Update attributes for a dequeue */
    void dequeue_update_attr();

    /*! \brief Wait until the batch at the position index is free.
     *
     * Note: index can be updated, that is why it is a reference
     * \param index reference to the index of the batch to lock
     * \return The index where it is currently locked
     */
    uint wait_and_lock(const std::atomic<uint>& index)
    {
        uint tmp_index;
        while (true)
        {
            tmp_index = index.load();
            if (batch_mutexes_[tmp_index].try_lock())
                break;
        }
        return tmp_index;
    }

  private: /* Private attributes */
    cuda_tools::UniquePtr<char> data_{nullptr};

    /*! \brief FastUpdatesHolder entry */
    FastUpdatesHolder<QueueType>::Value fast_updates_entry_;

    /*! \brief The current number of frames in the queue
     *
     * This variable must always be equal to
     * batch_size_ * size_ + curr_batch_counter
     */
    std::atomic<uint>& curr_nb_frames_;
    /*! \brief The total number of frames that can be contained in the queue according to batch size
     *
     * With respect to batch size (batch_size_ * max_size_)
     */
    std::atomic<uint>& total_nb_frames_;
    /*! \brief The total number of frames that can be contained in the queue */
    std::atomic<uint> frame_capacity_{0};

  public:
    /*! \brief Current number of full batches
     *
     * Can concurrently be modified by the producer (enqueue) and the consumer (dequeue, resize)
     */
    std::atomic<uint> size_{0};
    /*! \brief Number of frames in a batch
     *
     * Batch size can only be changed by the consumer when the producer is
     * blocked. Thus std::atomic is not required.
     */
    uint batch_size_{0};
    /*! \brief Max number of batch of frames in the queue
     *
     * Max size can only be changed by the consumer when the producer is
     * blocked. Thus std:atomic is not required.
     */
    uint max_size_{0};

  private:
    /*! \brief Start batch index. */
    std::atomic<uint> start_index_{0};
    /*! \brief End index is the index after the last batch */
    std::atomic<uint> end_index_{0};
    /*! \brief Number of batches. Batch size can only be changed by the consumer */
    std::atomic<bool> has_overwritten_{false};

    /*! \brief Counting how many frames have been enqueued in the current batch. */
    std::atomic<uint> curr_batch_counter_{0};

    /*! \name Synchronization attributes
     * \{
     */
    mutable std::mutex m_producer_busy_;
    std::unique_ptr<std::mutex[]> batch_mutexes_{nullptr};
    std::unique_ptr<cudaStream_t[]> batch_streams_{nullptr};
    /*! \} */

    /*!
     * \brief Whether the queue is on the GPU or not (and if data is a CudaUniquePtr or a GPUUniquePtr)
     *
     */
    std::atomic<Device>& device_;
};
} // namespace holovibes
