#include "queue.hh"
#include "tools_conversion.cuh"

#include "info_manager.hh"
#include <stdexcept>

namespace holovibes
{
  using guard = std::lock_guard<std::mutex>;

  Queue::Queue(const camera::FrameDescriptor& frame_desc, const unsigned int elts, std::string name)
    : frame_desc_(frame_desc)
    , size_(frame_desc_.frame_size())
    , pixels_(frame_desc_.frame_res())
    , max_elts_(elts)
    , curr_elts_(0)
    , start_(0)
	, display_(true)
    , is_big_endian_(frame_desc.depth >= 2 &&
    frame_desc.endianness == camera::BIG_ENDIAN)
    , name_(name)
  {
    if (cudaMalloc(&buffer_, size_ * elts) != CUDA_SUCCESS)
    {
      std::cerr << "Queue: couldn't allocate queue" << std::endl;
      throw std::logic_error(name_ + ": couldn't allocate queue");
    }

    frame_desc_.endianness = camera::LITTLE_ENDIAN;
    cudaStreamCreate(&stream_);
  }

  Queue::~Queue()
  {
	  if (display_)
    gui::InfoManager::remove_info_safe(name_);
    if (cudaFree(buffer_) != CUDA_SUCCESS)
      std::cerr << "Queue: couldn't free queue" << std::endl;
    cudaStreamDestroy(stream_);
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

  unsigned int Queue::get_max_elts() const
  {
    return max_elts_;
  }

  void* Queue::get_start()
  {
    guard guard(mutex_);
    return buffer_ + start_ * size_;
  }

  unsigned int Queue::get_start_index()
  {
    return start_;
  }

  void* Queue::get_end()
  {
    guard guard(mutex_);
    return buffer_ + ((start_ + curr_elts_) % max_elts_) * size_;
  }

  void* Queue::get_last_images(const unsigned n)
  {
    guard guard(mutex_);
    return buffer_ + ((start_ + curr_elts_ - n) % max_elts_) * size_;
  }

  unsigned int Queue::get_end_index()
  {
    guard guard(mutex_);
    return (start_ + curr_elts_) % max_elts_;
  }

  bool Queue::enqueue(void* elt, cudaMemcpyKind cuda_kind)
  {
    guard guard(mutex_);

    const unsigned int end_ = (start_ + curr_elts_) % max_elts_;
    char* new_elt_adress = buffer_ + (end_ * size_);
    cudaError_t cuda_status = cudaMemcpyAsync(new_elt_adress,
      elt,
      size_,
      cuda_kind,
      stream_);

    if (cuda_status != CUDA_SUCCESS)
    {
      std::cerr << "Queue: couldn't enqueue" << std::endl;
	  if (display_)
      gui::InfoManager::update_info_safe(name_, "couldn't enqueue");
      return false;
    }
    if (is_big_endian_)
      endianness_conversion((unsigned short*)new_elt_adress, (unsigned short*)new_elt_adress, frame_desc_.frame_res(), stream_);

    if (curr_elts_ < max_elts_)
      ++curr_elts_;
    else
      start_ = (start_ + 1) % max_elts_;
	if (display_)
    gui::InfoManager::update_info_safe(name_,
      std::to_string(curr_elts_) + std::string("/") + std::to_string(max_elts_));
    return true;
  }

  void Queue::dequeue(void* dest, cudaMemcpyKind cuda_kind)
  {
    guard guard(mutex_);

    if (curr_elts_ > 0)
    {
      void* first_img = buffer_ + start_ * size_;
      cudaMemcpyAsync(dest, first_img, size_, cuda_kind, stream_);
      start_ = (start_ + 1) % max_elts_;
      --curr_elts_;
	  if (display_)
      gui::InfoManager::update_info_safe(name_,
        std::to_string(curr_elts_) + std::string("/") + std::to_string(max_elts_));
    }
  }

  void Queue::dequeue()
  {
    guard guard(mutex_);

    if (curr_elts_ > 0)
    {
      start_ = (start_ + 1) % max_elts_;
      --curr_elts_;
    }
  }

  void Queue::flush()
  {
    guard guard(mutex_);

    curr_elts_ = 0;
    start_ = 0;
  }

  void Queue::set_display(bool value)
  {
	  display_ = value;
  }
}