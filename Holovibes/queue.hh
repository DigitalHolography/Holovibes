#pragma once

# include <iostream>
# include <mutex>
# include <cuda.h>
# include <cuda_runtime.h>

# include <frame_desc.hh>

namespace holovibes
{
  /*! \brief Queue class is a custom circular FIFO data structure. It can handle
  ** CPU or GPU data. This class is used to store the raw images, provided
  ** by the camera, and holograms.
  **
  ** This Queue is thread safe, it is impossible to enqueue and dequeue simultaneously.
  ** As well as it is impossible to get a value from the class getters while another
  ** object is enqueuing or dequeuing.
  **
  ** The Queue ensure that all elements it contains are written in little endian endianness.
  **
  */
  class Queue
  {
  public:
    /*! \brief Queue constructor
    **
    ** Please note that every size used in internal allocations for the Queue depends
    ** on provided FrameDescriptor, i-e in frame_size() and frame_res() methods.
    **
    ** Please note that when you allocate a Queue, its element number elts should be at least greater
    ** by 2 that what you need (e-g: 10 elements Queue should be allocated with a elts of 12).
    **
    ** \param frame_desc Either the FrameDescriptor of the camera that provides
    ** images or a FrameDescriptor used for computations.
    ** \param elts Max number of elements that the queue can contain.
    **/
    Queue(const camera::FrameDescriptor& frame_desc, const unsigned int elts);

    /*! \brief Queue destructor */
    ~Queue();

    /*! \return the size of one frame (i-e element) of the Queue in bytes. */
    size_t get_size() const;

    /*! \return pointer to internal buffer that contains data. */
    void* get_buffer();

    /*! \return FrameDescriptor of the Queue */
    const camera::FrameDescriptor& get_frame_desc() const;

    /*! \return the size of one frame (i-e element) of the Queue in pixels. */
    int get_pixels();

    /*! \return the number of elements the Queue currently contains.
    **  As this is the most used method, it is inlined here.
    */
    size_t get_current_elts() const
    {
      return curr_elts_;
    }

    /*! \return the number of elements the Queue can contains at its maximum. */
    unsigned int get_max_elts() const;

    /*! \return pointer to first frame. */
    void* get_start();

    /*! \return index of first frame (as the Queue is circular, it is not always zero). */
    unsigned int get_start_index();

    /*! \return pointer right after last frame */
    void* get_end();

    /*! \return pointer to end_index - n frame */
    void* get_last_images(const unsigned n);

    /*! \return index of the frame right after the last one containing data */
    unsigned int get_end_index();

    /*! \brief Enqueue method
    **
    ** Copies the given elt according to cuda_kind cuda memory type, then convert
    ** to little endian if the camera is in big endian.
    **
    ** If the maximum element number has been reached, the Queue overwrite the first frame.
    **
    ** \param elt pointer to element to enqueue
    ** \param cuda_kind kind of memory transfer (e-g: CudaMemCpyHostToDevice ...)
    */
    bool enqueue(void* elt, cudaMemcpyKind cuda_kind);

    /*! \brief Dequeue method overload
    **
    ** Copy the first element of the Queue into dest according to cuda_kind
    ** cuda memory type then update internal attributes.
    **
    ** \param dest destination of element copy
    ** \param cuda_kind kind of memory transfer (e-g: CudaMemCpyHostToDevice ...)
    */
    void dequeue(void* dest, cudaMemcpyKind cuda_kind);

    /*! \brief Dequeue method
    **
    ** Update internal attributes (reduce Queue current elements and change start pointer)
    */
    void dequeue();

    /*! Flushes the Queue */
    void flush();

  private:
    /*! FrameDescriptor of the Queue */
    camera::FrameDescriptor frame_desc_;

    /*! Size of one element in bytes */
    const size_t size_;

    /*! Size of one element in pixels */
    const int pixels_;

    /*! Maximum elements number */
    const unsigned int max_elts_;

    /*! Current elements number */
    size_t curr_elts_;

    /*! Start index */
    unsigned int start_;

    /*! Boolean used to know if camera is writing frames in big endian endianness */
    const bool is_big_endian_;

    /*! Data buffer */
    char* buffer_;

    /*! Mutex for critical code sections (threads safety) */
    std::mutex mutex_;

    cudaStream_t  stream_;
  };
}