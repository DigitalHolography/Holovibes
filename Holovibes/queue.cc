#include "queue.hh"

namespace queue
{
  bool Queue::enqueue(void* elt)
  {
    unsigned int end_ = (start_ + curr_elts_) % max_elts_;
    memcpy(buffer_ + (end_ * size_), elt, size_);

    if (curr_elts_ < max_elts_)
      ++curr_elts_;
    else
      start_ = (start_ + 1) % max_elts_;

    return true;
  }

  void* Queue::dequeue()
  {
    return dequeue(1);
  }

  void* Queue::dequeue(unsigned int elts_nb)
  {
    if (elts_nb <= curr_elts_)
    {
      void* old_ptr = buffer_ + start_ * size_;
      start_ = (start_ + elts_nb) % max_elts_;
      curr_elts_ -= elts_nb;

      return old_ptr;
    }
    else
      return nullptr;
  }

  // debug only
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
}