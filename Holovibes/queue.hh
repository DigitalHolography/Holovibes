#ifndef QUEUE_HH
# define QUEUE_HH

# include <cstdlib>
# include <iostream>
# include <mutex>
# include <cuda.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>

namespace holovibes
{
  class Queue
  {
  public:
    Queue()
    {
    }

    Queue(unsigned int size, unsigned int elts)
      : size_(size),
      max_elts_(elts),
      curr_elts_(0),
      start_(0)
    {
      if (cudaMalloc(&buffer_, size * elts) != CUDA_SUCCESS)
        std::cerr << "Queue: couldn't allocate queue" << std::endl;
    }

    ~Queue()
    {
      if(cudaFree(buffer_) != CUDA_SUCCESS)
        std::cerr << "Queue: couldn't free queue" << std::endl;
    }

    size_t get_size() const
    {
      return size_;
    }

    void* get_buffer()
    {
      return buffer_;
    }

    size_t get_current_elts();
    unsigned int get_max_elts() const;
    void* get_start();
    unsigned int get_start_index();
    void* get_end();
    void* get_last_images(int n);
    unsigned int get_end_index();

    bool enqueue(void* elt);
    void* dequeue();
    void* dequeue(size_t elts_nb);

    // debug only
    void print() const;

  private:
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