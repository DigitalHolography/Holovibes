#include "stdafx.h"
#include "thread_compute.hh"

namespace holovibes
{
  ThreadCompute::ThreadCompute(unsigned int p,
    unsigned int images_nb,
    float lambda,
    float z,
    Queue& input_q) :
    p_(p),
    images_nb_(images_nb),
    lambda_(lambda),
    z_(z),
    input_q_(input_q),
    compute_on_(true),
    thread_(&ThreadCompute::thread_proc, this)
  {
    camera::FrameDescriptor fd = input_q_.get_frame_desc();
    fd.depth = 2;
    fd.endianness = camera::LITTLE_ENDIAN;
    output_q_ = new Queue(fd, input_q_.get_max_elts());
  }

  ThreadCompute::~ThreadCompute()
  {
    compute_on_ = false;

    if (thread_.joinable())
      thread_.join();

    delete output_q_;
    cudaFree(lens_);
    cudaFree(output_buffer_);
    cudaFree(sqrt_vec_);
  }

  Queue& ThreadCompute::get_queue()
  {
    return *output_q_;
  }

  void ThreadCompute::compute_hologram()
  {
    if (input_q_.get_current_elts() >= images_nb_)
    {
      fft_1(images_nb_, &input_q_, lens_, sqrt_vec_, output_buffer_, plan_);
      unsigned short *shifted = output_buffer_ + p_ * input_q_.get_pixels();
      shift_corners(&shifted, output_q_->get_frame_desc().width, output_q_->get_frame_desc().height);
      output_q_->enqueue(shifted, cudaMemcpyDeviceToDevice);
      input_q_.dequeue();
    }
  }

  void ThreadCompute::thread_proc()
  {
    cudaMalloc(&output_buffer_, input_q_.get_pixels() * sizeof(unsigned short) * images_nb_);

    sqrt_vec_ = make_sqrt_vec(65536);

    lens_ = create_lens(input_q_.get_frame_desc().width, input_q_.get_frame_desc().height, lambda_, z_);

    cufftPlan3d(&plan_, images_nb_, input_q_.get_frame_desc().width, input_q_.get_frame_desc().height, CUFFT_C2C);

    while (compute_on_)
      compute_hologram();
  }
}