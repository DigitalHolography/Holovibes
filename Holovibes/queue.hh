#ifndef QUEUE_HH
# define QUEUE_HH

# include <cstdlib>
# include <iostream>

namespace holovibes
{
  class Queue
  {
  public:
    Queue(unsigned int size, unsigned int elts)
      : size_(size),
      max_elts_(elts),
      curr_elts_(0),
      start_(0)
    {
      buffer_ = (char*)malloc(size * elts);
    }

    ~Queue()
    {
      free(buffer_);
    }

    size_t get_size() const
    {
      return size_;
    }

    size_t get_current_elts() const
    {
      return curr_elts_;
    }

    unsigned int get_max_elts() const
    {
      return max_elts_;
    }

    void* get_start() const
    {
      return buffer_ + start_ * size_;
    }

    unsigned int get_start_index() const
    {
      return start_;
    }

    void* get_end() const
    {
      return buffer_ + ((start_ + curr_elts_) % max_elts_) * size_;
    }

    unsigned int get_end_index() const
    {
      return (start_ + curr_elts_) % max_elts_;
    }

    bool enqueue(void* elt);
    void* dequeue();
    void* dequeue(size_t elts_nb);

    // debug only
    void print() const;

  private:
    size_t size_;
    unsigned int max_elts_;
    size_t curr_elts_;
    unsigned int start_;
    char* buffer_;
  };
}

#endif /* !QUEUE_HH */