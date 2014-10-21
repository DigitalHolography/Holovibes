#include "stdafx.h"
#include "queue.hh"

namespace holovibes
{
  size_t Queue::get_current_elts()
  {
    mutex_.lock();
    unsigned int curr_elts = curr_elts_;
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

  bool Queue::enqueue(void* elt)
  {
    mutex_.lock();

    unsigned int end_ = (start_ + curr_elts_) % max_elts_;
    memcpy(buffer_ + (end_ * size_), elt, size_);

    if (curr_elts_ < max_elts_)
      ++curr_elts_;
    else
      start_ = (start_ + 1) % max_elts_;

    mutex_.unlock();
    return true;
  }

  void Queue::dequeue(void* dest)
  {
    mutex_.lock();
    if (curr_elts_ > 0)
    {
      void* last_img = buffer_ + ((start_ + curr_elts_ - 1) % max_elts_) * size_;
      memcpy(dest, last_img, size_);
    }
    mutex_.unlock();
  }

  /*
  void* Queue::dequeue(size_t elts_nb)
  {
    if (elts_nb <= curr_elts_)
    {
      mutex_.lock();

      void* old_ptr = buffer_ + start_ * size_;
      start_ = (start_ + elts_nb) % max_elts_;
      curr_elts_ -= elts_nb;

      mutex_.unlock();

      return old_ptr;
    }
    else
      return nullptr;
  }
  */

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