#include "queue.hh"
#include "tools_conversion.cuh"

namespace holovibes
{
  Queue::Queue(const camera::FrameDescriptor& frame_desc, unsigned int elts)
    : frame_desc_(frame_desc)
    , size_(frame_desc_.frame_size())
    , pixels_(frame_desc_.frame_res())
    , max_elts_(elts)
    , curr_elts_(0)
    , start_(0)
    , is_big_endian_(frame_desc.depth == 2 &&
    frame_desc.endianness == camera::BIG_ENDIAN)
  {
    if (cudaMalloc(&buffer_, size_ * elts) != CUDA_SUCCESS)
      std::cerr << "Queue: couldn't allocate queue" << std::endl;

    frame_desc_.endianness = camera::LITTLE_ENDIAN;
  }

  Queue::~Queue()
  {
    if (cudaFree(buffer_) != CUDA_SUCCESS)
      std::cerr << "Queue: couldn't free queue" << std::endl;
  }

  size_t Queue::get_size() const
  {
    return size_;
  }

  void* Queue::get_buffer()
  {
    return buffer_;
  }

  const camera::FrameDescriptor& Queue::get_frame_desc() const
  {
    return frame_desc_;
  }

  int Queue::get_pixels()
  {
    return pixels_;
  }

  size_t Queue::get_current_elts()
  {
    mutex_.lock();
    size_t curr_elts = curr_elts_;
    mutex_.unlock();
    return curr_elts;
  }

  unsigned int Queue::get_max_elts() const
  {
    return max_elts_;
  }

  void* Queue::get_start()
  {
    mutex_.lock();
    void* start_ptr = buffer_ + start_ * size_;
    mutex_.unlock();
    return start_ptr;
  }

  unsigned int Queue::get_start_index()
  {
    mutex_.lock();
    unsigned int start_index = start_;
    mutex_.unlock();
    return start_index;
  }

  void* Queue::get_end()
  {
    mutex_.lock();
    void* end_ptr = buffer_ + ((start_ + curr_elts_) % max_elts_) * size_;
    mutex_.unlock();
    return end_ptr;
  }

  void* Queue::get_last_images(int n)
  {
    mutex_.lock();
    void* end_ptr = buffer_ + ((start_ + curr_elts_ - n) % max_elts_) * size_;
    mutex_.unlock();
    return end_ptr;
  }

  unsigned int Queue::get_end_index()
  {
    mutex_.lock();
    unsigned int end_index = (start_ + curr_elts_) % max_elts_;
    mutex_.unlock();
    return end_index;
  }

  bool Queue::enqueue(void* elt, cudaMemcpyKind cuda_kind)
  {
    mutex_.lock();

    unsigned int end_ = (start_ + curr_elts_) % max_elts_;
    char* new_elt_adress = buffer_ + (end_ * size_);
    int cuda_status = cudaMemcpy(new_elt_adress,
      elt,
      size_,
      cuda_kind);

    if (cuda_status != CUDA_SUCCESS)
    {
      std::cerr << "Queue: couldn't enqueue" << std::endl;
      mutex_.unlock();
      return false;
    }
    if (is_big_endian_)
      endianness_conversion((unsigned short*)new_elt_adress, (unsigned short*)new_elt_adress, frame_desc_.frame_res());

    if (curr_elts_ < max_elts_)
      ++curr_elts_;
    else
      start_ = (start_ + 1) % max_elts_;

    mutex_.unlock();
    return true;
  }

  void Queue::dequeue(void* dest, cudaMemcpyKind cuda_kind)
  {
    mutex_.lock();
    if (curr_elts_ > 0)
    {
      void* first_img = buffer_ + start_ * size_;
      cudaMemcpy(dest, first_img, size_, cuda_kind);
      start_ = (start_ + 1) % max_elts_;
      curr_elts_--;
    }
    mutex_.unlock();
  }

  void Queue::dequeue()
  {
    mutex_.lock();
    if (curr_elts_ > 0)
    {
      start_ = (start_ + 1) % max_elts_;
      curr_elts_ -= 1;
    }
    mutex_.unlock();
  }

  void Queue::flush()
  {
    mutex_.lock();
    curr_elts_ = 0;
    start_ = 0;
    mutex_.unlock();
  }
}