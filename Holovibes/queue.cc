#include "stdafx.h"
#include "queue.hh"

namespace holovibes
{
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

    if (frame_desc_.depth == 2 && frame_desc_.endianness == camera::BIG_ENDIAN)
      endianness_conversion((unsigned short*)new_elt_adress, (unsigned short*)new_elt_adress, size_);

    if (cuda_status != CUDA_SUCCESS)
      std::cerr << "Queue: couldn't enqueue" << std::endl;

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
      void* last_img = buffer_ + ((start_ + curr_elts_ - 1) % max_elts_) * size_;
      cudaMemcpy(dest, last_img, size_, cuda_kind);
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

#if _DEBUG
  void Queue::print() const
  {
    unsigned int end_ = (start_ + curr_elts_) % max_elts_;

    std::cout << "-- Queue --" << std::endl;
    std::cout << "start: " << start_
      << " end: " << end_
      << " elts: " << curr_elts_ << std::endl;
    for (unsigned int i = 0; i < max_elts_; ++i)
    {
      std::cout << (int)(*(buffer_ + i * size_)) << std::endl;
    }
  }
#endif
}