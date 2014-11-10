#include "stdafx.h"
#include "thread_compute.hh"

namespace holovibes
{
  ThreadCompute::ThreadCompute(unsigned int p,
    unsigned int images_nb,
    float lambda,
    float z,
    Queue& input_q,
    int type) :
    p_(p),
    images_nb_(images_nb),
    lambda_(lambda),
    z_(z),
    input_q_(input_q),
    compute_on_(true),
    thread_(&ThreadCompute::thread_proc, this),
    type_(type)
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
      if (type_ == 1)
      {
         fft_1(images_nb_, &input_q_, lens_, sqrt_vec_, output_buffer_, plan3d_);
        unsigned short *shifted = output_buffer_ + p_ * input_q_.get_pixels();
        shift_corners(&shifted, output_q_->get_frame_desc().width, output_q_->get_frame_desc().height);
        output_q_->enqueue(shifted, cudaMemcpyDeviceToDevice);
        input_q_.dequeue();
      }
       else
      {
      fft_2(images_nb_, &input_q_, lens_, sqrt_vec_, output_buffer_, plan3d_, p_, plan2d_);
      unsigned short *shifted = output_buffer_;
      shift_corners(&shifted, output_q_->get_frame_desc().width, output_q_->get_frame_desc().height);
      output_q_->enqueue(shifted, cudaMemcpyDeviceToDevice);
      input_q_.dequeue();
      }
    }
  }

  void ThreadCompute::thread_proc()
  {
    if (type_ == 1)
      cudaMalloc(&output_buffer_, input_q_.get_pixels() * sizeof(unsigned short) * images_nb_);
    else
      cudaMalloc(&output_buffer_, input_q_.get_pixels() * sizeof(unsigned short));

    sqrt_vec_ = make_sqrt_vec(65536);

    if (type_ == 1)
      lens_ = create_lens(input_q_.get_frame_desc(), lambda_, z_);
    else
      lens_ = create_spectral(lambda_, z_, input_q_.get_frame_desc().width, input_q_.get_frame_desc().height, 1.0e-6f, 1.0e-6f,input_q_.get_frame_desc());
     
    std::cout << type_ << std::endl;
      cufftPlan3d(&plan3d_, images_nb_, input_q_.get_frame_desc().width, input_q_.get_frame_desc().height, CUFFT_C2C);
      cufftPlan2d(&plan2d_, input_q_.get_frame_desc().width, input_q_.get_frame_desc().height, CUFFT_C2C);


    while (compute_on_)
      compute_hologram();
  }
}