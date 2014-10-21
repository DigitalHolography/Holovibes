#ifndef QUEUE_HH
# define QUEUE_HH

# include <cstdlib>
# include <iostream>
# include <mutex>
# include "frame_desc.hh"

namespace holovibes
{
  class Queue
  {
  public:
    Queue(const camera::FrameDescriptor& frame_desc, unsigned int elts)
      : frame_desc_(frame_desc)
      , size_(frame_desc.frame_size())
      , max_elts_(elts)
      , curr_elts_(0)
      , start_(0)
    {
      buffer_ = (char*)malloc(size_ * elts);
    }

    ~Queue()
    {
      free(buffer_);
    }

    size_t get_size() const
    {
      return size_;
    }

    size_t get_current_elts();
    unsigned int get_max_elts() const;
    void* get_start();
    unsigned int get_start_index();
    void* get_end();
    void* get_last_images(int n);
    unsigned int get_end_index();

    bool enqueue(void* elt);
    void dequeue(void* dest);
    //void* dequeue(size_t elts_nb);

    // debug only
    void print() const;

  private:
    // Frame descriptor
    const camera::FrameDescriptor& frame_desc_;

    // Size of one element
    size_t size_;

    // Maximum elements number
    unsigned int max_elts_;

    // Current elements number
    size_t curr_elts_;

    // Start index
    unsigned int start_;

    // Data buffer
    char* buffer_;

    // Mutex for critical code sections (threads safety)
    std::mutex mutex_;
  };
}

#endif /* !QUEUE_HH */