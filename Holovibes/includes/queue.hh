/*! \file
 *
 * \brief Implementation of a circular queue
 *
 * Queue class is a custom circular FIFO data structure. It can handle
 * CPU or GPU data. This class is used to store the raw images, provided
 * by the camera, and holograms.
 */
#pragma once

#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "batch_input_queue.hh"
#include "display_queue.hh"
#include "global_state_holder.hh"
namespace holovibes
{
/*! \class Queue
 *
 * \brief Circular queue to handle CPU and GPU data
 *
 * Queue class is a custom circular FIFO data structure. It can handle
 * CPU or GPU data. This class is used to store the raw images, provided
 * by the camera, and holograms.
 *
 * This Queue is thread safe, it is impossible to enqueue and dequeue
 * simultaneously.
 * As well as it is impossible to get a value from the class getters while
 * another
 * object is enqueuing or dequeuing.
 *
 * The Queue ensures that all elements it contains are written in little endian.
 */
class Queue final : public DisplayQueue
{
    friend class BatchInputQueue;

  public:
    using MutexGuard = std::lock_guard<std::mutex>;

  public:
    /*! \brief Queue constructor
     *
     * Please note that every size used in internal allocations for the Queue depends
     * on provided FrameDescriptor, i-e in frame_size() and frame_res() methods.
     *
     * \param fd The frame descriptor representing frames stored in the queue
     * \param max_size The max size of the queue
     * \param type The type of the queue
     */
    Queue(const FrameDescriptor& fd,
          const unsigned int max_size,
          QueueType type = QueueType::UNDEFINED,
          unsigned int input_width = 0,
          unsigned int input_height = 0,
          unsigned int bytes_per_pixel = 1);

    /*! \brief Destructor of the queue */
    ~Queue();

    /*! \name Getters
     * \{
     */

    /*! \return Pointer to internal buffer that contains data. */
    void* get_data() const { return data_; }

    /*! \return The number of elements the Queue currently contains. */
    unsigned int get_size() const { return size_; }

    /*! \return The number of elements the Queue can contains at its maximum. */
    unsigned int get_max_size() const { return max_size_; }

    /*! \return Pointer to first frame. */
    void* get_start() const { return data_.get() + start_index_ * fd_.get_frame_size(); }

    /*! \return Index of first frame (as the Queue is circular, it is not always zero). */
    unsigned int get_start_index() const { return start_index_; }

    /*! \return Pointer right after last frame */
    void* get_end() const { return data_.get() + ((start_index_ + size_) % max_size_) * fd_.get_frame_size(); }

    /*! \return Pointer to the last image */
    void* get_last_image() const override
    {
        MutexGuard mGuard(mutex_);
        // if the queue is empty, return a random frame
        return data_.get() + ((start_index_ + size_ - 1) % max_size_) * fd_.get_frame_size();
    }

    /*! \return Index of the frame right after the last one containing data */
    unsigned int get_end_index() const { return (start_index_ + size_) % max_size_; }

    /*! \return Getter to the queue mutex */
    std::mutex& get_guard() { return mutex_; }
    /*! \} */

    /*! \return If queue has overridden at least a frame during an enqueue */
    bool has_overridden() const { return has_overridden_; }

    /*! \name Methods
     * \{
     */
    /*! \brief Empty the Queue and change its size.
     *
     * \param size The new size of the Queue
     * \param stream
     */
    void resize(const unsigned int size, const cudaStream_t stream);

    /*! \brief Enqueue method
     *
     * Copies the given elt according to cuda_kind cuda memory type, then convert
     * to little endian if the camera is in big endian.
     *
     * If the maximum element number has been reached, the Queue overwrite the first frame.
     *
     * The memcpy are synch for ??.
     *
     * \param elt Pointer to element to enqueue
     * \param stream
     * \param cuda_kind Kind of memory transfer (e-g: CudaMemCpyHostToDevice...)
     */
    bool enqueue(void* elt, const cudaStream_t stream, cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    /*! \brief Copy elements (no dequeue) and enqueue in dest.
     *
     * Batch copy method
     *
     * \param dest Output queue
     * \param nb_elts Number of elements to add in the queue
     * \param stream
     */
    void copy_multiple(Queue& dest, unsigned int nb_elts, const cudaStream_t stream);

    /*! \brief Enqueue method for multiple elements
     *
     * Batch enqueue method
     *
     * The memcpy are async
     *
     * \param elts List of elements to add in the queue
     * \param nb_elts Number of elements to add in the queue
     * \param stream
     * \param cuda_kind Kind of memory transfer (e-g: CudaMemCpyHostToDevice...)
     * \return The success of the operation: False if an error occurs
     */
    bool enqueue_multiple(void* elts,
                          unsigned int nb_elts,
                          const cudaStream_t stream,
                          cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    void enqueue_from_48bit(void* src, const cudaStream_t stream, cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    /*! \brief Dequeue method overload
     *
     * Copy the first element of the Queue into dest according to cuda_kind
     * cuda memory type then update internal attributes.
     *
     * \param dest Destination of element copy
     * \param stream
     * \param cuda_kind Kind of memory transfer (e-g: CudaMemCpyHostToDevice...)
     */
    void dequeue(void* dest, const cudaStream_t stream, cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    /*! \brief Dequeue method
     *
     * Update internal attributes
     * Decrease the size of the queue and change start pointer
     * Lock the queue
     *
     * \param nb_elt The number of elements to dequeue
     */
    void dequeue(const unsigned int nb_elts = 1);

    /*! \brief Dequeue method without mutex
     *
     * Update internal attributes
     * Decrease the size of the queue and change start pointer
     *
     * \param nb_elt The number of elements to dequeue
     */
    void dequeue_non_mutex(const unsigned int nb_elts = 1);

    /*! \brief Empties the Queue. */
    void clear();

    /*! \return Check if the queue is full */
    bool is_full() const;

    /*! \return String containing the buffer size in MB */
    std::string calculate_size(void) const;
    /*! \} */

  private:
    /*! \brief Auxiliary method of enqueue multiple.
     *
     * Mostly make the copy
     *
     * \param out The output buffer in which the frames are copied
     * \param in The input buffer from which the frames are copied
     * \param nb_elts The number of elements to enqueue
     * \param stream
     * \param cuda_kind Kind of memory transfer (e-g: CudaMemCpyHostToDevice...)
     */
    void enqueue_multiple_aux(
        void* out, void* in, unsigned int nb_elts, const cudaStream_t stream, cudaMemcpyKind cuda_kind);

    // Forward declaration
    struct QueueRegion;

    /*! \brief Auxiliary method of copy multiple.
     *
     * Make the async copy
     *
     * \param src Queue region info of the source queue
     * \param dst Queue region info of the dst queue
     * \param frame_size Size of the frame in bytes
     * \param stream Stream perfoming the copy
     */
    static void
    copy_multiple_aux(QueueRegion& src, QueueRegion& dst, const size_t frame_size, const cudaStream_t stream);

  private: /* Attributes */
    /*! \brief Mutex to lock the queue */
    mutable std::mutex mutex_;

    /*! \brief Size of the queue (number of frames currently stored in the queue)
     *
     * This attribute is atomic because it is required by the wait frames function.
     * A thread is enqueueing a frame, meanwhile the other thread is waiting
     * for a specific size of the queue. Using an atomic avoid locking the queue.
     * This is only used by the concurrent queue. However, it is needed to be declare in the regular queue.
     */
    std::atomic<uint> size_;
    /*! \brief Maximum size of the queue (capacity) */
    uint max_size_;

    /* \} */

    /*! \brief Type of the queue */
    QueueType type_;

    /*! \brief The index of the first frame in the queue */
    unsigned int start_index_;
    const bool is_big_endian_;
    /*! \brief The actual buffer in which the frames are stored */
    cuda_tools::UniquePtr<char> data_;

    // Utils used for square input mode
    /*! \brief Original width of the input */
    unsigned int input_width_;
    /*! \brief Original height of the input */
    unsigned int input_height_;
    /*! \brief number of byte(s) to encode a pixel */
    unsigned int bytes_per_pixel;

    /*! \brief Wheter frames have been overridden during an enqueue. */
    bool has_overridden_;

  private:
    /*! \struct QueueRegion
     *
     * \brief Struct to represents a region in the queue, or two regions in case of overflow.
     *
     * First is the first region
     * Second is the second region if overflow, nullptr otherwise.
     * In case of overflow, this struct will look like
     * |----------------- (start_index_) ---------------|
     * |	    	second          |         first         |
     */
    struct QueueRegion
    {
        char* first = nullptr;
        char* second = nullptr;
        unsigned int first_size = 0;
        unsigned int second_size = 0;

        bool overflow(void) { return second != nullptr; }

        void consume_first(unsigned int size, size_t frame_size)
        {
            first += static_cast<size_t>(size) * frame_size * sizeof(char);
            first_size -= size;
        }

        void consume_second(unsigned int size, size_t frame_size)
        {
            second += static_cast<size_t>(size) * frame_size * sizeof(char);
            second_size -= size;
        }
    };
};
} // namespace holovibes
