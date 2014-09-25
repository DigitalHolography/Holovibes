#ifndef QUEUE_HH
# define QUEUE_HH

# include <cstdlib>
# include <iostream>

namespace queue
{
  class Queue
  {
  public:
    Queue(size_t size, unsigned int elts)
      : size_(size),
      elts_(elts),
      end_(0)
    {
      buffer_ = (char*)malloc(size * elts);
    }

    ~Queue()
    {
      free(buffer_);
    }

    unsigned int get_size() const
    {
      return size_;
    }

    unsigned int get_max_elts() const
    {
      return elts_;
    }

    bool enqueue(void* elt);
    void* dequeue();
    void* dequeue(unsigned int elts_nb);

    // debug only
    void print() const;

  private:
    size_t size_;
    unsigned int elts_;
    unsigned int end_;
    char* buffer_;
  };
}

#endif /* !QUEUE_HH */