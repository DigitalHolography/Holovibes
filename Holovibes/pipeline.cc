#include "stdafx.h"
#include "pipeline.hh"

#include <cassert>
#include "fft1.cuh"
#include "fft2.cuh"
#include "tools.cuh"
#include "preprocessing.cuh"

namespace holovibes
{
  Pipeline::Pipeline(
    Queue& input,
    Queue& output,
    ComputeDescriptor& desc)
    : fn_vect_()
    , compute_desc_(desc)
    , input_(input)
    , output_(output)
    , gpu_input_buffer_(nullptr)
    , gpu_output_buffer_(nullptr)
    , gpu_float_buffer_(nullptr)
    , gpu_sqrt_vector_(nullptr)
    , gpu_lens_(nullptr)
    , plan3d_(0)
    , plan2d_(0)
    , gpu_input_frame_ptr_(nullptr)
    , autofocus_requested_(false)
    , autocontrast_requested_(false)
    , refresh_requested_(false)
    , update_n_requested_(false)
  {
    const unsigned short nsamples = desc.nsamples;

    /* gpu_input_buffer */
    cudaMalloc<cufftComplex>(&gpu_input_buffer_,
      sizeof(cufftComplex) * input_.get_pixels() * nsamples);

    /* gpu_output_buffer */
    cudaMalloc<unsigned short>(&gpu_output_buffer_,
      sizeof(unsigned short) * input_.get_pixels());

    /* gpu_float_buffer */
    cudaMalloc<float>(&gpu_float_buffer_,
      sizeof(float) * input_.get_pixels());

    /* Square root vector */
    cudaMalloc<float>(&gpu_sqrt_vector_, sizeof(float) * 65535);
    make_sqrt_vect(gpu_sqrt_vector_, 65535);

    /* gpu_lens */
    cudaMalloc(&gpu_lens_,
      input_.get_pixels() * sizeof(cufftComplex));

    /* CUFFT plan3d */
    cufftPlan3d(
      &plan3d_,
      nsamples,                       // NX
      input_.get_frame_desc().width,  // NY
      input_.get_frame_desc().height, // NZ
      CUFFT_C2C);

    /* CUFFT plan2d */
    cufftPlan2d(
      &plan2d_,
      input_.get_frame_desc().width,
      input_.get_frame_desc().height,
      CUFFT_C2C);

    refresh();
  }

  Pipeline::~Pipeline()
  {
    /* CUFFT plan2d */
    cufftDestroy(plan2d_);

    /* CUFFT plan3d */
    cufftDestroy(plan3d_);
    
    /* gpu_lens */
    cudaFree(gpu_lens_);

    /* Square root vector */
    cudaFree(gpu_sqrt_vector_);

    /* gpu_float_buffer */
    cudaFree(gpu_float_buffer_);

    /* gpu_output_buffer */
    cudaFree(gpu_output_buffer_);

    /* gpu_input_buffer */
    cudaFree(gpu_input_buffer_);
  }

  void Pipeline::update_n_parameter(unsigned short n)
  {
    /* CUFFT plan3d realloc */
    cufftDestroy(plan3d_);
    cufftPlan3d(
      &plan3d_,
      n,                              // NX
      input_.get_frame_desc().width,  // NY
      input_.get_frame_desc().height, // NZ
      CUFFT_C2C);

    /* gpu_input_buffer realloc */
    cudaFree(gpu_input_buffer_);
    gpu_input_buffer_ = nullptr;
    cudaMalloc<cufftComplex>(&gpu_input_buffer_,
      sizeof(cufftComplex) * input_.get_pixels() * n);
  }

  void Pipeline::refresh()
  {
    const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
    const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

    /* Clean current vector. */
    fn_vect_.clear();

    if (update_n_requested_)
    {
      update_n_requested_ = false;
      update_n_parameter(compute_desc_.nsamples);
    }

    // Fill input complex buffer.
    fn_vect_.push_back(std::bind(
      make_contiguous_complex,
      std::ref(input_),
      gpu_input_buffer_,
      compute_desc_.nsamples.load(),
      gpu_sqrt_vector_));

    if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
    {
      // Initialize FFT1 lens.
      fft1_lens(
        gpu_lens_,
        input_fd,
        compute_desc_.lambda,
        compute_desc_.zdistance);

      // Add FFT1.
      fn_vect_.push_back(std::bind(
        fft_1,
        gpu_input_buffer_,
        gpu_lens_,
        plan3d_,
        input_fd.frame_res(),
        compute_desc_.nsamples.load()));

      gpu_input_frame_ptr_ = gpu_input_buffer_ + compute_desc_.pindex * input_fd.frame_res();
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      fft2_lens(
        gpu_lens_,
        input_fd,
        compute_desc_.lambda,
        compute_desc_.zdistance);

      gpu_input_frame_ptr_ = gpu_input_buffer_ + compute_desc_.pindex * input_fd.frame_res();

      fn_vect_.push_back(std::bind(
        fft_2,
        gpu_input_buffer_,
        gpu_input_frame_ptr_,
        gpu_lens_,
        plan3d_,
        plan2d_,
        input_fd.frame_res(),
        compute_desc_.nsamples.load(),
        compute_desc_.pindex.load()));
    }
    else
      assert(!"Impossible case.");

    /* Apply conversion to unsigned short. */
    if (compute_desc_.view_mode == ComputeDescriptor::MODULUS)
    {
      fn_vect_.push_back(std::bind(
        complex_to_modulus,
        gpu_input_frame_ptr_,
        gpu_float_buffer_,
        input_fd.frame_res()));
    }
    else if (compute_desc_.view_mode == ComputeDescriptor::SQUARED_MODULUS)
    {
      fn_vect_.push_back(std::bind(
        complex_to_squared_modulus,
        gpu_input_frame_ptr_,
        gpu_float_buffer_,
        input_fd.frame_res()));
    }
    else if (compute_desc_.view_mode == ComputeDescriptor::ARGUMENT)
    {
      fn_vect_.push_back(std::bind(
        complex_to_argument,
        gpu_input_frame_ptr_,
        gpu_float_buffer_,
        input_fd.frame_res()));
    }
    else
      assert(!"Impossible case.");

    /* [POSTPROCESSING] Everything behind this line uses output_frame_ptr */

    if (compute_desc_.log_scale_enabled)
    {
      fn_vect_.push_back(std::bind(
        apply_log10,
        gpu_float_buffer_,
        input_fd.frame_res()));
    }

    fn_vect_.push_back(std::bind(
      float_to_ushort,
      gpu_float_buffer_,
      gpu_output_buffer_,
      input_fd.frame_res()));

    if (compute_desc_.shift_corners_enabled)
    {
      fn_vect_.push_back(std::bind(
        shift_corners,
        gpu_output_buffer_,
        output_fd.width,
        output_fd.height));
    }

    if (autofocus_requested_)
    {
      autofocus_requested_ = false;
      // push autofocus();
    }

    if (autocontrast_requested_)
    {
      autocontrast_requested_ = false;
      // push autocontrast();
    }

    if (autocontrast_requested_ ||
      autofocus_requested_)
    {
      request_refresh();
    }
  }

  void Pipeline::exec()
  {
    if (input_.get_current_elts() >= compute_desc_.nsamples)
    {
      for (FnVector::const_iterator cit = fn_vect_.cbegin();
        cit != fn_vect_.cend();
        ++cit)
        (*cit)();

      output_.enqueue(
        gpu_output_buffer_,
        cudaMemcpyDeviceToDevice);
      input_.dequeue();

      if (refresh_requested_)
      {
        refresh();
        refresh_requested_ = false;
      }
    }
  }

  void Pipeline::request_refresh()
  {
    refresh_requested_ = true;
  }

  void Pipeline::request_autocontrast()
  {
    autocontrast_requested_ = true;
    request_refresh();
  }

  void Pipeline::request_autofocus()
  {
    autofocus_requested_ = true;
    request_refresh();
  }

  void Pipeline::request_update_n(unsigned short n)
  {
    update_n_requested_ = true;
    compute_desc_.nsamples = n;
    request_refresh();
  }
}