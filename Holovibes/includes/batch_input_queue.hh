#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <atomic>
#include <mutex>

#include "queue.hh"

using uint = unsigned int;

namespace holovibes
{

/* Conditons:
** 2 threads: 1 Consumer (dequeue, copy multiple) and 1 producer
** batch size in the queue must be a submultiple of the queue size
** The order of operations cannot be guaranteed if the queue is full
** i.g. Enqueue, Dequeue, Enqueue might be processed in this order Enqueue,
** Dequeue, Enqueue
*/
class BatchInputQueue
{
  public: /* Public methods */
    BatchInputQueue(const uint total_nb_frames, const uint batch_size, const uint frame_size);

    ~BatchInputQueue();

    /*! \brief Enqueue a frame in the queue
    ** Called by the producer.
    ** The producer is in the critical while enqueueing in a batch
    ** and exit this critical section when a batch of frames is full
    ** in order to let the resize occure if needed.
    */
    void enqueue(const void* const input_frame,
                 const cudaMemcpyKind memcpy_kind = cudaMemcpyDeviceToDevice);

    // /!\ HOLO: Copy multiple will be using a regular queue in holovibes
    // But it cannot be tested easily here. Use an input queue just for
    // compilation testing
    /*! \brief Copy multiple
    ** Called by the consumer.
    */
    void copy_multiple(BatchInputQueue& dest);

    //! \brief Function used when dequeuing a batch of frame
    // src, dst, batch_size, frame_size, stream -> void
    using dequeue_func_t = std::function<void(const void* const, void* const,
      const uint, const uint, const cudaStream_t)>;

    /*! \brief Deqeue a batch of frames. Block until the queue has at least a
    ** full batch of frame.
    ** Called by the consumer.
    ** The queue must have at least a batch of frames filled.
    */
    void BatchInputQueue::dequeue(void* const dest, dequeue_func_t func);

    /*! \brief Resize with a new batch size
    ** Called by the consumer.
    ** Empty the queue.
    */
    void resize(const uint new_batch_size);

    /*! \brief Stop the producer.
    ** Exit the critical section of the producer.
    ** It is required because the producer can only exit the critical section
    ** only when a batch is full. However, the producer will stop its enqueues
    ** (in Holovibes the reader is stopped) and will not be able to exit the
    ** critical section in case the current batch is still not full.
    ** The queue needs this critical section to be exited for its destruction
    ** later.
    */
    void stop_producer();

    inline bool is_empty() const;

    inline uint get_size() const;

    inline bool has_overridden() const;

    // HOLO: Can it be removed?
    inline const void* get_data() const;

    inline uint get_frame_size() const;

  private: /* Private methods */

    /*! \brief Set size attributes and create mutexes and streams arrays.
    ** Used by the consumer and constructor
    ** \param total_nb_frames The maximum capacity of the queue in terms of number of frames
    ** (new_batch_size * max_size)
    ** \param new_batch_size The new number of frames in a batch
    */
    void create_mutexes_streams(const uint total_nb_frames, const uint new_batch_size);

    /*! \brief Destroy mutexes and streams arrays.
    ** Used by the consumer and constructor.
    */
    void destroy_mutexes_streams();

    /*! \brief Update queue indexes to make the queue empty
    ** Used by the consumer.
    */
    void make_empty();

  private: /* Private attributes */
    // HOLO: cuda_tools::UniquePtr
    char* data_;

    //! Size of a frame (number of pixels * depth) in bytes. Never modified.
    const uint frame_size_;

    /*! Current number of full batches
    ** Can concurrently be modified by the producer (enqueue)
    ** and the consumer (dequeue, resize)
    */
    std::atomic<uint> size_{0};
    /*! Number of frames in a batch
    ** Batch size can only be changed by the consumer when the producer is
    ** blocked. Thus std:atomic is not required.
    */
    uint batch_size_{0};
    /*! Max number of batch of frames in the queue
    ** Batch size can only be changed by the consumer when the producer is
    ** blocked. Thus std:atomic is not required.
    */
    uint max_size_{0};
    //! Start batch index.
    std::atomic<uint> start_index_{0};
    //! End index is the index after the last batch
    std::atomic<uint> end_index_{0};
    //! Number of batches. Batch size can only be changed by the consumer
    std::atomic<bool> has_overridden_{false};

    //! Couting how many frames have been enqueued in the current batch.
    std::atomic<uint> curr_batch_counter_{0};

    /* Synchronization attributes */
    std::mutex m_producer_busy_;
    std::unique_ptr<std::mutex[]> batch_mutexes_{nullptr};
    std::unique_ptr<cudaStream_t[]> batch_streams_{nullptr};
};
} // namespace holovibes