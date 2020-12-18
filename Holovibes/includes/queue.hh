/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Queue class is a custom circular FIFO data structure. It can handle
 * CPU or GPU data. This class is used to store the raw images, provided
 * by the camera, and holograms.
 *
 */
#pragma once

#include "frame_desc.hh"
#include "compute_descriptor.hh"
#include "unique_ptr.hh"

namespace holovibes
{
/*! \brief Queue class is a custom circular FIFO data structure. It can handle
** CPU or GPU data. This class is used to store the raw images, provided
** by the camera, and holograms.
**
** This Queue is thread safe, it is impossible to enqueue and dequeue
*simultaneously.
** As well as it is impossible to get a value from the class getters while
*another
** object is enqueuing or dequeuing.
**
** The Queue ensures that all elements it contains are written in little endian.
*/
class Queue
{
  public:
    using MutexGuard = std::lock_guard<std::mutex>;

    enum class QueueType
    {
        UNDEFINED,
        INPUT_QUEUE,
        OUTPUT_QUEUE,
        RECORD_QUEUE,
    };

  public:
    /*! \brief Queue constructor
    **
    ** Please note that every size used in internal allocations for the Queue
    *depends
    ** on provided FrameDescriptor, i-e in frame_size() and frame_res() methods.
    **
    ** \param fd The frame descriptor representing frames stored in the queue
    ** \param max_size The max size of the queue
    ** \param type The type of the queue
    **/
    Queue(const camera::FrameDescriptor& fd,
          const unsigned int max_size,
          QueueType type = QueueType::UNDEFINED,
          unsigned int input_width = 0,
          unsigned int input_height = 0,
          unsigned int bytes_per_pixel = 1);

    /*! \brief Destructor of the queue */
    ~Queue();

    /* Getters */
    /*! \return the size of one frame (i-e element) of the Queue in bytes. */
    inline size_t get_frame_size() const;

    /*! \return pointer to internal buffer that contains data. */
    inline void* get_data() const;

    /*! \return FrameDescriptor of the Queue */
    inline const camera::FrameDescriptor& get_fd() const;

    /*! \return the size of one frame (i-e element) of the Queue in pixels. */
    inline size_t get_frame_res() const;

    /*! \return the number of elements the Queue currently contains. */
    inline unsigned int get_size() const;

    /*! \return the number of elements the Queue can contains at its maximum. */
    inline unsigned int get_max_size() const;

    /*! \return pointer to first frame. */
    inline void* get_start() const;

    /*! \return index of first frame (as the Queue is circular, it is not always
     * zero). */
    inline unsigned int get_start_index() const;

    /*! \return pointer right after last frame */
    inline void* get_end() const;

    /*! \return pointer to the last image */
    inline void* get_last_image() const;

    /*! \return index of the frame right after the last one containing data */
    inline unsigned int get_end_index() const;

    /*! \return getter to the queue mutex */
    inline std::mutex& get_guard();

    /* Setters */
    /*! \brief Set the input mode (cropped, no modification, padding) */
    inline void set_square_input_mode(SquareInputMode mode);

    /*! \return if queue has overridden at least a frame during an enqueue */
    inline bool has_overridden() const;

    /* Methods */
    /*! \brief Empty the Queue and change its size.
    **
    ** \param size the new size of the Queue
    ** \param stream
    */
    void resize(const unsigned int size, const cudaStream_t stream);

    /*! \brief Enqueue method
    **
    ** Copies the given elt according to cuda_kind cuda memory type, then
    *convert
    ** to little endian if the camera is in big endian.
    **
    ** If the maximum element number has been reached, the Queue overwrite the
    *first frame.
    **
    ** The memcpy are synch for Qt
    **.
    ** \param elt pointer to element to enqueue
    ** \param stream
    ** \param cuda_kind kind of memory transfer (e-g: CudaMemCpyHostToDevice
    *...)
    */
    bool enqueue(void* elt,
                 const cudaStream_t stream,
                 cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    /*! \brief Copy method for multiple elements
    **
    **	Batch copy method
    **
    ** \param dest Output queue
    ** \param nb_elts Number of elements to add in the queue
    ** \param stream
    */
    void
    copy_multiple(Queue& dest, unsigned int nb_elts, const cudaStream_t stream);

    /*! \brief Enqueue method for multiple elements
    **
    ** Batch enqueue method
    **
    ** The memcpy are async
    **
    ** \param elts List of elements to add in the queue
    ** \param nb_elts Number of elements to add in the queue
    ** \param stream
    ** \param cuda_kind kind of memory transfer (e-g: CudaMemCpyHostToDevice
    *...)
    **
    ** \return The success of the operation: False if an error occurs
    */
    bool enqueue_multiple(void* elts,
                          unsigned int nb_elts,
                          const cudaStream_t stream,
                          cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    void Queue::enqueue_from_48bit(
        void* src,
        const cudaStream_t stream,
        cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    /*! \brief Dequeue method overload
    **
    ** Copy the first element of the Queue into dest according to cuda_kind
    ** cuda memory type then update internal attributes.
    **
    ** \param dest destination of element copy
    ** \param stream
    ** \param cuda_kind kind of memory transfer (e-g: CudaMemCpyHostToDevice
    *...)
    */
    void dequeue(void* dest,
                 const cudaStream_t stream,
                 cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

    /*! \brief Dequeue method
    **
    ** Update internal attributes
    ** Decrease the size of the queue and change start pointer
    ** Lock the queue
    ** \param nb_elt The number of elements to dequeue
    */
    void dequeue(const unsigned int nb_elts = 1);

    /*! \brief Dequeue method without mutex
    **
    ** Update internal attributes
    ** Decrease the size of the queue and change start pointer
    ** \param nb_elt The number of elements to dequeue
    */
    void dequeue_non_mutex(const unsigned int nb_elts = 1);

    /*! \brief Empties the Queue. */
    void clear();

    /*! \return check if the queue is full */
    bool is_full() const;

    /*! \return string containing the buffer size in MB*/
    std::string calculate_size(void) const;

  private: /* Private Methods */
    /*! \brief auxiliary method of enqueue multiple.
    ** Mostly make the copy
    ** \param out the output buffer in which the frames are copied
    ** \param in the input buffer from which the frames are copied
    ** \param nb_elts The number of elements to enqueue
    ** \param stream
    ** \param cuda_kind kind of memory transfer (e-g: CudaMemCpyHostToDevice
    *...)
    */
    void enqueue_multiple_aux(void* out,
                              void* in,
                              unsigned int nb_elts,
                              const cudaStream_t stream,
                              cudaMemcpyKind cuda_kind);

  private: /* Attributes */
    /*! \brief mutex to lock the queue */
    mutable std::mutex mutex_;
    /*! \brief frame descriptor of a frame store in the queue */
    camera::FrameDescriptor fd_;

    /*! \brief frame size from the frame descriptor */
    const size_t frame_size_;
    /*! \brief frame resolution from the frame descriptor */
    const size_t frame_res_;
    /*! \brief Maximum size of the queue (capacity) */
    std::atomic<unsigned int> max_size_;

    //! Type of the queue
    Queue::QueueType type_;

    /*! \brief Size of the queue (number of frames currently stored in the
    ** queue)
    ** This attribute is atomic because it is required by the wait frames
    ** function.
    ** A thread is enqueueing a frame, meanwhile the other thread is waiting
    ** for a specific size of the queue. Using an atomic avoid locking the
    ** queue.
    ** This is only used by the concurrent queue. However, it is needed to
    ** be declare in the regular queue.
    */
    std::atomic<unsigned int> size_;

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
    /*! \brief Input mode (NO_MODIFICATION, ZERO_PADDED_SQUARE,
    ** CROPPED_SQUARE)
    */
    SquareInputMode square_input_mode_;

    bool has_overridden_;
};

/*! \brief Struct to represents a region in the queue, or two regions in
** case of overflow.
** first is the first region
** second is the second region if overflow, nulpptr otherwise.
** In case of overflow, this struct will look like
** |----------------- (start_index_) ---------------|
** |		second          |         first         |
*/
struct QueueRegion
{
    char* first = nullptr;
    char* second = nullptr;
    unsigned int first_size = 0;
    unsigned int second_size = 0;

    bool overflow(void) { return second != nullptr; }

    void consume_first(unsigned int size, unsigned int frame_size)
    {
        first += size * frame_size;
        first_size -= size;
    }

    void consume_second(unsigned int size, unsigned int frame_size)
    {
        second += size * frame_size;
        second_size -= size;
    }
};
} // namespace holovibes

#include "queue.hxx"