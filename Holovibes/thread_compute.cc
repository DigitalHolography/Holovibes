#include "stdafx.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>

#include "thread_compute.hh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "preprocessing.cuh"

namespace holovibes
{
  ThreadCompute::ThreadCompute(
    const ComputeDescriptor& desc,
    Queue& input_q)
    : compute_desc_(desc)
    , input_q_(input_q)
    , compute_on_(true)
    , thread_(&ThreadCompute::thread_proc, this)
  {
    camera::FrameDescriptor fd = input_q_.get_frame_desc();
    fd.depth = 2;

    output_q_ = new Queue(fd, input_q_.get_max_elts());
  }

  ThreadCompute::~ThreadCompute()
  {
    compute_on_ = false;

    if (thread_.joinable())
      thread_.join();

    delete output_q_;
  }

  Queue& ThreadCompute::get_queue()
  {
    return *output_q_;
  }

  void ThreadCompute::thread_proc()
  {
    /* Ressources allocation */
    float* sqrt_array = make_sqrt_vect(65536);
    /* Output buffer containing p images ordered in frequency. */
    unsigned short *pbuffer = nullptr;
    if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
    {
    cudaMalloc(
      &pbuffer,
        input_q_.get_pixels() * sizeof(unsigned short)* compute_desc_.nsamples);
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      cudaMalloc(
        &pbuffer,
        input_q_.get_pixels() * sizeof(unsigned short));
    }
    cufftHandle plan3d;
    cufftPlan3d(
      &plan3d,
      compute_desc_.nsamples,            // NX
      input_q_.get_frame_desc().width,   // NY
      input_q_.get_frame_desc().height,  // NZ
      CUFFT_C2C);

    cufftHandle plan2d;
    if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      cufftPlan2d(&plan2d, input_q_.get_frame_desc().width,
        input_q_.get_frame_desc().height,
        CUFFT_C2C);
    }


    cufftComplex* lens = nullptr;


    if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
    {
      lens = create_lens(
        input_q_.get_frame_desc(),
        compute_desc_.lambda,
        compute_desc_.zdistance);
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      lens = create_spectral(
        compute_desc_.lambda,
        compute_desc_.zdistance,
        input_q_.get_frame_desc().width,
        input_q_.get_frame_desc().height,
        input_q_.get_frame_desc().pixel_size,
        input_q_.get_frame_desc().pixel_size,input_q_.get_frame_desc());
    }
    else
      assert(!"Impossible case");

    /* Thread loop */
    while (compute_on_)
    {
      if (input_q_.get_current_elts() >= compute_desc_.nsamples)
      {
        if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
        {
          fft_1(
          compute_desc_.nsamples,
          &input_q_,
          lens,
          sqrt_array,
          pbuffer,
            plan3d);

        /* Shifting */
        unsigned short *shifted = pbuffer + compute_desc_.pindex * input_q_.get_pixels();
        shift_corners(&shifted, output_q_->get_frame_desc().width, output_q_->get_frame_desc().height);
        /* Store p-th image */
        output_q_->enqueue(shifted, cudaMemcpyDeviceToDevice);
        }
        else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
        {
          fft_2(compute_desc_.nsamples, &input_q_, lens, sqrt_array, pbuffer, plan3d, compute_desc_.pindex, plan2d);
          shift_corners(&pbuffer, output_q_->get_frame_desc().width, output_q_->get_frame_desc().height);
          //img2disk("ak.raw", pbuffer, input_q_.get_pixels() * sizeof (unsigned short));
          //exit(0);
          output_q_->enqueue(pbuffer, cudaMemcpyDeviceToDevice);
        }

        
        input_q_.dequeue();
      }
    }

    /* Free ressources */
    cudaFree(lens);
    cufftDestroy(plan2d);
    cufftDestroy(plan3d);
    cudaFree(pbuffer);
    cudaFree(sqrt_array);
  }
}