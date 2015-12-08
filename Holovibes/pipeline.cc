#include <algorithm>
#include "tools.hh"

#include "pipeline.hh"

#include "fft1.cuh"
#include "fft2.cuh"
#include "stft.cuh"
#include "tools.cuh"
#include "tools_conversion.cuh"
#include "preprocessing.cuh"
#include "contrast_correction.cuh"
#include "vibrometry.cuh"
#include "autofocus.cuh"

namespace holovibes
{
  Pipeline::Pipeline(
    Queue& input,
    Queue& output,
    ComputeDescriptor& desc)
    : ICompute(input, output, desc)
  {
    /* The 16-bit buffer is allocated (and deallocated) separately,
     as no Module is associated directly to it. */
    cudaMalloc<unsigned short>(&gpu_short_buffer_,
      sizeof(unsigned short)* input_.get_pixels());

    refresh();
  }

  Pipeline::~Pipeline()
  {
    stop_pipeline();
    cudaFree(gpu_short_buffer_);
  }

  void Pipeline::stop_pipeline()
  {
    delete_them(modules_, [](Module* module) { delete module; });
    delete_them(gpu_complex_buffers_, [](cufftComplex* buffer) { cudaFree(buffer); });
    delete_them(gpu_float_buffers_, [](float* buffer) { cudaFree(buffer); });
    gpu_pindex_buffers_.clear();
  }

  void Pipeline::exec()
  {
    while (!termination_requested_)
    {
      if (input_.get_current_elts() >= compute_desc_.nsamples.load())
      {
        // Say to each Module that there is work to be done.
        std::for_each(modules_.begin(),
          modules_.end(),
          [](Module* module) { module->task_done_ = false; });

        while (std::any_of(modules_.begin(),
          modules_.end(),
          [](Module *module) { return !module->task_done_; }))
        {
          continue;
        }

        // Now that everyone is finished, rotate datasets as seen by the Modules.
        step_forward();

        input_.dequeue();
        output_.enqueue(
          gpu_short_buffer_,
          cudaMemcpyDeviceToDevice);

        if (refresh_requested_)
          refresh();
      }
    }
  }

  template <class T>
  Module* Pipeline::create_module(std::list<T*>& gpu_buffers, size_t buf_size)
  {
    T             *buffer = nullptr;

    cudaMalloc(&buffer, sizeof(T)* buf_size); gpu_buffers.push_back(buffer);

    return ();
  }

  void Pipeline::refresh()
  {
    cufftComplex                  *gpu_complex_buffer = nullptr;
    float                         *gpu_float_buffer = nullptr;
    const camera::FrameDescriptor &input_fd = input_.get_frame_desc();

    refresh_requested_ = false;
    stop_pipeline();

    if (update_n_requested_)
    {
      update_n_requested_ = false;
      update_n_parameter(compute_desc_.nsamples);
    }

    if (abort_construct_requested_)
    {
      std::cout << "[PIPELINE] abort_construct_requested" << std::endl;
      return;
    }

    modules_.push_back(new Module()); // C1
    modules_.push_back(new Module()); // C2
    modules_.push_back(new Module()); // F1

    cudaMalloc(&gpu_complex_buffer, sizeof(cufftComplex)* input_.get_pixels() * input_length_);
    gpu_complex_buffers_.push_back(gpu_complex_buffer);
    cudaMalloc(&gpu_complex_buffer, sizeof(cufftComplex)* input_.get_pixels() * input_length_);
    gpu_complex_buffers_.push_back(gpu_complex_buffer);
    cudaMalloc(&gpu_float_buffer, sizeof(float)* input_.get_pixels());
    gpu_float_buffers_.push_back(gpu_float_buffer);
    cudaMalloc(&gpu_float_buffer, sizeof(float)* input_.get_pixels());
    gpu_float_buffers_.push_back(gpu_float_buffer);

    std::for_each(gpu_complex_buffers_.begin(),
      gpu_complex_buffers_.end(),
      [&](cufftComplex* buf) { gpu_pindex_buffers_.push_back(buf + compute_desc_.pindex * input_fd.frame_res()); });

    if (autofocus_requested_)
    {
      // LOL ...
    }

    modules_[0]->add_worker(std::bind(
      make_contiguous_complex,
      std::ref(input_),
      std::ref(gpu_complex_buffers_[0]),
      compute_desc_.nsamples.load(),
      gpu_sqrt_vector_,
      modules_[0]->stream_
      ));

    if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
    {
      // Initialize FFT1 lens.
      fft1_lens(
        gpu_lens_,
        input_fd,
        compute_desc_.lambda,
        compute_desc_.zdistance);

      // Add FFT1.
      modules_[0]->add_worker(std::bind(
        fft_1,
        std::ref(gpu_complex_buffers_[0]),
        gpu_lens_,
        plan3d_,
        input_fd.frame_res(),
        compute_desc_.nsamples.load(),
        modules_[0]->stream_
        ));
    }

    modules_[1]->add_worker(std::bind(
      complex_to_modulus,
      std::ref(gpu_pindex_buffers_[1]),
      std::ref(gpu_float_buffers_[0]),
      input_fd.frame_res(),
      modules_[1]->stream_
      ));

    modules_[2]->add_worker(std::bind(
      float_to_ushort,
      std::ref(gpu_float_buffers_[1]),
      gpu_short_buffer_,
      input_fd.frame_res(),
      modules_[2]->stream_
      ));
  }

  void Pipeline::step_forward()
  {
    if (gpu_complex_buffers_.size() > 1)
    {
      std::rotate(gpu_complex_buffers_.begin(),
        gpu_complex_buffers_.begin() + 1,
        gpu_complex_buffers_.end());
    }

    if (gpu_float_buffers_.size() > 1)
    {
      std::rotate(gpu_float_buffers_.begin(),
        gpu_float_buffers_.begin() + 1,
        gpu_float_buffers_.end());
    }

    if (gpu_pindex_buffers_.size() > 1)
    {
      std::rotate(gpu_pindex_buffers_.begin(),
        gpu_pindex_buffers_.begin() + 1,
        gpu_pindex_buffers_.end());
    }
  }

  void Pipeline::record_float()
  {
    // TODO
  }
}