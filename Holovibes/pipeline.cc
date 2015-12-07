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
    // TODO : Initialize modules by binding resources to std::functions.
    //        Allocate is_finished_ and set every value to false.
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
    delete_them(streams_, [](cudaStream_t& stream) { cudaStreamDestroy(stream); });
    delete_them(is_finished_, [](bool* is_finish) { delete is_finish; });
    delete_them(gpu_float_buffers_, [](float* buffer) { cudaFree(buffer); });
    delete_them(gpu_complex_buffers_, [](cufftComplex* buffer) { cudaFree(buffer); });
  }

  void Pipeline::exec()
  {
    while (!termination_requested_)
    {
      if (input_.get_current_elts() >= compute_desc_.nsamples.load())
      {
        std::for_each(is_finished_.begin(),
          is_finished_.end(),
          [](bool* is_finish) { *is_finish = false; });

        while (std::any_of(is_finished_.begin(),
          is_finished_.end(),
          [](bool *finish) { return !*finish; }))
        {
          continue;
        }

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

  void Pipeline::update_n_parameter(unsigned short n)
  {
    ICompute::update_n_parameter(n);
    // TODO
  }

  template <class T>
  Module* Pipeline::create_module(std::list<T*>& gpu_buffers, size_t buf_size)
  {
    T             *buffer = nullptr;
    cudaStream_t  stream;

    cudaMalloc(&buffer, sizeof(T)* buf_size);
    gpu_buffers.push_back(buffer);

    streams_.push_back(stream);
    cudaStreamCreate(&(streams_.back()));

    is_finished_.push_back(new bool(true));

    return (new Module(is_finished_.back()));
  }

  void Pipeline::refresh()
  {
    stop_pipeline();

    update_n_parameter(compute_desc_.nsamples);

    Module* module = nullptr;
    Module *moduleTmp = nullptr;
    const camera::FrameDescriptor& input_fd = input_.get_frame_desc();

    module = create_module<cufftComplex>(gpu_complex_buffers_, input_.get_pixels() * input_length_);
    // make_contiguous_complex
    // gen fft1_lens
    // fft_1

    module->add_worker(std::bind(
      make_contiguous_complex,
      std::ref(input_),
      std::ref(gpu_complex_buffers_.back()),
      compute_desc_.nsamples.load(),
      gpu_sqrt_vector_,
      streams_.back()
      ));

    // Initialize FFT1 lens.
    fft1_lens(
      gpu_lens_,
      input_fd,
      compute_desc_.lambda,
      compute_desc_.zdistance);
    module->add_worker(std::bind(
      fft_1,
      std::ref(gpu_complex_buffers_.back()),
      gpu_lens_,
      plan3d_,
      input_fd.frame_res(),
      compute_desc_.nsamples.load(),
      streams_.back()
      ));

    modules_.push_back(module);
    moduleTmp = create_module<cufftComplex>(gpu_complex_buffers_, input_.get_pixels() * input_length_);
    // complex_to_modulus (to_float)

    modules_.push_back(moduleTmp);

    module = create_module<float>(gpu_float_buffers_, input_.get_pixels());
    // float_to_ushort

    module->add_worker(std::bind(
      float_to_ushort,
      std::ref(gpu_float_buffers_.back()),
      gpu_short_buffer_,
      input_fd.frame_res(),
      streams_.back()
      ));

    modules_.push_back(module);

    module = create_module<float>(gpu_float_buffers_, input_.get_pixels());
    modules_.push_back(module);

    moduleTmp->add_worker(std::bind(
      complex_to_modulus,
      std::ref(gpu_complex_buffers_.back()),
      std::ref(gpu_float_buffers_.back()),
      input_fd.frame_res(),
      streams_.back()
      ));
  }

  void Pipeline::step_forward()
  {
    if (gpu_float_buffers_.size() > 1)
    {
      std::swap(gpu_float_buffers_.front(), gpu_float_buffers_.back());
      /*    std::rotate(gpu_float_buffers_.begin(),
            gpu_float_buffers_.begin() + 1,
            gpu_float_buffers_.end());
            */
    }

    if (gpu_complex_buffers_.size() > 1)
    {
      std::swap(gpu_complex_buffers_.front(), gpu_complex_buffers_.back());
    }
    /*
    std::rotate(gpu_complex_buffers_.begin(),
    gpu_complex_buffers_.begin() + 1,
    gpu_complex_buffers_.end());
    */
  }

  void Pipeline::record_float()
  {
    // TODO
  }
}