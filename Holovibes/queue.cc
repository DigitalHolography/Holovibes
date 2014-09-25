#include "queue.hh"

namespace queue
{
  bool Queue::enqueue(void* elt)
  {
    //memcpy
    memcpy(buffer_ + (end_ * size_), elt, size_);
    end_ = (end_ + 1) % elts_;

    return true;
  }

  void* Queue::dequeue()
  {
    //FIXME
    return nullptr;
  }

  void* Queue::dequeue(unsigned int elts_nb)
  {
    //FIXME
    return nullptr;
  }

  // debug only
  void Queue::print() const
  {
    std::cout << "-- Queue --" << std::endl;
    for (unsigned int i = 0; i < elts_; ++i)
    {
      std::cout << (int)(*(buffer_ + i * size_)) << std::endl;
    }
  }
}