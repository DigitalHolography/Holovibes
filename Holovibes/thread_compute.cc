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
    output_q_ = new Queue(fd, input_q_.get_max_elts());
    cudaMalloc(&output_buffer_, input_q_.get_pixels() * sizeof(unsigned short) * images_nb_);

    if (input_q_.get_frame_desc().depth > 1)
      sqrt_vec_ = make_sqrt_vec(65536);
    else
      sqrt_vec_ = make_sqrt_vec(256);

    lens_ = create_lens(input_q_.get_frame_desc().width, input_q_.get_frame_desc().height, lambda_, z_);
    cufftPlan3d(&plan_, images_nb_, input_q_.get_frame_desc().width, input_q_.get_frame_desc().height, CUFFT_C2C);
  }

  ThreadCompute::~ThreadCompute()
  {
    compute_on_ = false;

    delete output_q_;
    cudaFree(lens_);
    cudaFree(output_buffer_);
    cudaFree(sqrt_vec_);

    if (thread_.joinable())
      thread_.join();
  }

  void ThreadCompute::compute_hologram()
  {
    std::cout << "compute hologram" << std::endl;
    fft_1(images_nb_, &input_q_, lens_, sqrt_vec_, output_buffer_, plan_);
    output_q_->enqueue(output_buffer_ + p_ * input_q_.get_pixels(), cudaMemcpyDeviceToDevice);
  }

  void ThreadCompute::thread_proc()
  {
    while (input_q_.get_current_elts() >= images_nb_ && compute_on_)
      compute_hologram();
  }
}