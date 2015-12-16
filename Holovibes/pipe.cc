#include "pipe.hh"

#include <cassert>
#include <algorithm>

#include "fft1.cuh"
#include "fft2.cuh"
#include "stft.cuh"
#include "tools.cuh"
#include "tools_conversion.cuh"
#include "preprocessing.cuh"
#include "contrast_correction.cuh"
#include "vibrometry.cuh"

namespace holovibes
{
  Pipe::Pipe(
    Queue& input,
    Queue& output,
    ComputeDescriptor& desc)
    : ICompute(input, output, desc)
    , fn_vect_()
    , gpu_input_buffer_(nullptr)
    , gpu_output_buffer_(nullptr)
    , gpu_float_buffer_(nullptr)
    , gpu_input_frame_ptr_(nullptr)
  {
    /* gpu_input_buffer */
    cudaMalloc<cufftComplex>(&gpu_input_buffer_,
      sizeof(cufftComplex)* input_.get_pixels() * input_length_);
    /* gpu_output_buffer */
    cudaMalloc<unsigned short>(&gpu_output_buffer_,
      sizeof(unsigned short)* input_.get_pixels());

    /* gpu_float_buffer */
    cudaMalloc<float>(&gpu_float_buffer_,
      sizeof(float)* input_.get_pixels());

    // Setting the cufft plans to work on the default stream.
    cufftSetStream(plan1d_, static_cast<cudaStream_t>(0));
    cufftSetStream(plan2d_, static_cast<cudaStream_t>(0));
    cufftSetStream(plan3d_, static_cast<cudaStream_t>(0));

    refresh();
  }

  Pipe::~Pipe()
  {
    /* gpu_float_buffer */
    cudaFree(gpu_float_buffer_);

    /* gpu_output_buffer */
    cudaFree(gpu_output_buffer_);

    /* gpu_input_buffer */
    cudaFree(gpu_input_buffer_);
  }

  void Pipe::update_n_parameter(unsigned short n)
  {
    ICompute::update_n_parameter(n);

    /* gpu_input_buffer */
    cudaDestroy<cudaError_t>(&(gpu_input_buffer_));
    cudaMalloc<cufftComplex>(&gpu_input_buffer_,
      sizeof(cufftComplex)* input_.get_pixels() * input_length_);
  }

  void Pipe::refresh()
  {
    ICompute::refresh();
    /* As the Pipe uses a single CUDA stream for its computations,
     * we have to explicitly use the default stream (0).
     * Because std::bind does not allow optional parameters to be
     * deduced and bound, we have to use static_cast<cudaStream_t>(0) systematically. */

    const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
    const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

    refresh_requested_ = false;
    /* Clean current vector. */
    fn_vect_.clear();

    if (update_n_requested_)
    {
      update_n_requested_ = false;
      update_n_parameter(compute_desc_.nsamples);
    }

    if (abort_construct_requested_)
      return;

    if (!autofocus_requested_)
    {
      // Fill input complex buffer, one frame at a time.
      fn_vect_.push_back(std::bind(
        make_contiguous_complex,
        std::ref(input_),
        gpu_input_buffer_,
        input_length_,
        gpu_sqrt_vector_,
        static_cast<cudaStream_t>(0)));
    }
    else
    {
      autofocus_init();

      fn_vect_.push_back(std::bind(
        &Pipe::cudaMemcpyNoReturn,
        this,
        gpu_input_buffer_,
        af_env_.gpu_input_buffer_tmp,
        af_env_.gpu_input_size,
        cudaMemcpyDeviceToDevice));
    }

    if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
    {
      // Initialize FFT1 lens.
      if (!autofocus_requested_)
      {
        fft1_lens(
          gpu_lens_,
          input_fd,
          compute_desc_.lambda,
          compute_desc_.zdistance,
          static_cast<cudaStream_t>(0));
      }
      else
      {
        fn_vect_.push_back(std::bind(
          fft1_lens,
          gpu_lens_,
          input_fd,
          compute_desc_.lambda.load(),
          std::ref(af_env_.z),
          static_cast<cudaStream_t>(0)));
      }

      // Add FFT1.
      fn_vect_.push_back(std::bind(
        fft_1,
        gpu_input_buffer_,
        gpu_lens_,
        plan3d_,
        input_fd.frame_res(),
        compute_desc_.nsamples.load(),
        static_cast<cudaStream_t>(0)));

      /* p frame pointer */
      gpu_input_frame_ptr_ = gpu_input_buffer_ + compute_desc_.pindex * input_fd.frame_res();

      if (compute_desc_.vibrometry_enabled)
      {
        /* q frame pointer */
        cufftComplex* q = gpu_input_buffer_ + compute_desc_.vibrometry_q * input_fd.frame_res();

        fn_vect_.push_back(std::bind(
          frame_ratio,
          gpu_input_frame_ptr_,
          q,
          gpu_input_frame_ptr_,
          input_fd.frame_res(),
          static_cast<cudaStream_t>(0)));
      }
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      // Initialize FFT1 lens.
      if (!autofocus_requested_)
      {
        fft2_lens(
          gpu_lens_,
          input_fd,
          compute_desc_.lambda,
          compute_desc_.zdistance,
          static_cast<cudaStream_t>(0));
      }
      else
      {
        fn_vect_.push_back(std::bind(
          fft2_lens,
          gpu_lens_,
          input_fd,
          compute_desc_.lambda.load(),
          std::ref(af_env_.z),
          static_cast<cudaStream_t>(0)));
      }
      /* p frame pointer */
      gpu_input_frame_ptr_ = gpu_input_buffer_ + compute_desc_.pindex * input_fd.frame_res();

      if (compute_desc_.vibrometry_enabled)
      {
        fn_vect_.push_back(std::bind(
          fft_2,
          gpu_input_buffer_,
          gpu_lens_,
          plan3d_,
          plan2d_,
          input_fd.frame_res(),
          compute_desc_.nsamples.load(),
          compute_desc_.pindex.load(),
          compute_desc_.vibrometry_q.load(),
          static_cast<cudaStream_t>(0)));

        /* q frame pointer */
        cufftComplex* q = gpu_input_buffer_ + compute_desc_.vibrometry_q * input_fd.frame_res();

        fn_vect_.push_back(std::bind(
          frame_ratio,
          gpu_input_frame_ptr_,
          q,
          gpu_input_frame_ptr_,
          input_fd.frame_res(),
          static_cast<cudaStream_t>(0)));
      }
      else
      {
        fn_vect_.push_back(std::bind(
          fft_2,
          gpu_input_buffer_,
          gpu_lens_,
          plan3d_,
          plan2d_,
          input_fd.frame_res(),
          compute_desc_.nsamples.load(),
          compute_desc_.pindex.load(),
          compute_desc_.pindex.load(),
          static_cast<cudaStream_t>(0)));
      }
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::STFT)
    {
      // Initialize FFT1 lens.
      if (!autofocus_requested_)
      {
        fft1_lens(
          gpu_lens_,
          input_fd,
          compute_desc_.lambda,
          compute_desc_.zdistance,
          static_cast<cudaStream_t>(0));
      }
      else
      {
        fn_vect_.push_back(std::bind(
          fft1_lens,
          gpu_lens_,
          input_fd,
          compute_desc_.lambda.load(),
          std::ref(af_env_.z),
          static_cast<cudaStream_t>(0)));
      }

      curr_elt_stft_ = 0;
      // Add STFT.
      fn_vect_.push_back(std::bind(
        stft,
        gpu_input_buffer_,
        gpu_lens_,
        gpu_stft_buffer_,
        gpu_stft_dup_buffer_,
        plan2d_,
        plan1d_,
        compute_desc_.stft_roi_zone.load(),
        curr_elt_stft_,
        input_fd,
        compute_desc_.nsamples.load(),
        static_cast<cudaStream_t>(0)));

      fn_vect_.push_back(std::bind(
        stft_recontruct,
        gpu_input_buffer_,
        gpu_stft_dup_buffer_,
        compute_desc_.stft_roi_zone.load(),
        input_fd,
        (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_width() : input_fd.width),
        (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_height() : input_fd.height),
        compute_desc_.pindex.load(),
        compute_desc_.nsamples.load(),
        static_cast<cudaStream_t>(0)));

      /* frame pointer */
      gpu_input_frame_ptr_ = gpu_input_buffer_;

      if (compute_desc_.vibrometry_enabled)
      {
        /* q frame pointer */
        cufftComplex* q = q_gpu_stft_buffer_;

        fn_vect_.push_back(std::bind(
          stft_recontruct,
          q,
          gpu_stft_dup_buffer_,
          compute_desc_.stft_roi_zone.load(),
          input_fd,
          (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_width() : input_fd.width),
          (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_height() : input_fd.height),
          compute_desc_.vibrometry_q.load(),
          compute_desc_.nsamples.load(),
          static_cast<cudaStream_t>(0)));

        fn_vect_.push_back(std::bind(
          frame_ratio,
          gpu_input_frame_ptr_,
          q,
          gpu_input_frame_ptr_,
          input_fd.frame_res(),
          static_cast<cudaStream_t>(0)));
      }

      if (average_requested_)
      {
        if (compute_desc_.stft_roi_zone.load().area())
          fn_vect_.push_back(std::bind(
          &Pipe::average_stft_caller,
          this,
          gpu_stft_dup_buffer_,
          input_fd.width,
          input_fd.height,
          compute_desc_.stft_roi_zone.load().get_width(),
          compute_desc_.stft_roi_zone.load().get_height(),
          compute_desc_.signal_zone.load(),
          compute_desc_.noise_zone.load(),
          compute_desc_.nsamples.load(),
          static_cast<cudaStream_t>(0)));
        average_requested_ = false;
      }
    }
    else
      assert(!"Impossible case.");

    if (compute_desc_.unwrapping_enabled)
    {
      // Phase unwrapping
      fn_vect_.push_back(std::bind(
        unwrap,
        gpu_input_buffer_,
        input_fd.width,
        input_fd.height,
        compute_desc_.nsamples.load()));

      // Converting angle information in floating-point representation.
      fn_vect_.push_back(std::bind(
        complex_to_angle,
        gpu_input_frame_ptr_,
        gpu_float_buffer_,
        input_fd.frame_res(),
        static_cast<cudaStream_t>(0)));
    }
    else
    {
      /* Apply conversion to unsigned short. */
      if (compute_desc_.view_mode == ComputeDescriptor::MODULUS)
      {
        fn_vect_.push_back(std::bind(
          complex_to_modulus,
          gpu_input_frame_ptr_,
          gpu_float_buffer_,
          input_fd.frame_res(),
          static_cast<cudaStream_t>(0)));
      }
      else if (compute_desc_.view_mode == ComputeDescriptor::SQUARED_MODULUS)
      {
        fn_vect_.push_back(std::bind(
          complex_to_squared_modulus,
          gpu_input_frame_ptr_,
          gpu_float_buffer_,
          input_fd.frame_res(),
          static_cast<cudaStream_t>(0)));
      }
      else if (compute_desc_.view_mode == ComputeDescriptor::ARGUMENT)
      {
        fn_vect_.push_back(std::bind(
          complex_to_argument,
          gpu_input_frame_ptr_,
          gpu_float_buffer_,
          input_fd.frame_res(),
          static_cast<cudaStream_t>(0)));
      }
      else
        assert(!"Impossible case.");

      /* [POSTPROCESSING] Everything behind this line uses output_frame_ptr */

      if (compute_desc_.shift_corners_enabled)
      {
        fn_vect_.push_back(std::bind(
          shift_corners,
          gpu_float_buffer_,
          output_fd.width,
          output_fd.height,
          static_cast<cudaStream_t>(0)));
      }

      if (average_requested_)
      {
        if (average_record_requested_)
        {
          fn_vect_.push_back(std::bind(
            &Pipe::average_record_caller,
            this,
            gpu_float_buffer_,
            input_fd.width,
            input_fd.height,
            compute_desc_.signal_zone.load(),
            compute_desc_.noise_zone.load(),
            static_cast<cudaStream_t>(0)));

          average_record_requested_ = false;
        }
        else
        {
          fn_vect_.push_back(std::bind(
            &Pipe::average_caller,
            this,
            gpu_float_buffer_,
            input_fd.width,
            input_fd.height,
            compute_desc_.signal_zone.load(),
            compute_desc_.noise_zone.load(),
            static_cast<cudaStream_t>(0)));
        }

        average_requested_ = false;
      }

      if (compute_desc_.log_scale_enabled)
      {
        fn_vect_.push_back(std::bind(
          apply_log10,
          gpu_float_buffer_,
          input_fd.frame_res(),
          static_cast<cudaStream_t>(0)));
      }

      if (autocontrast_requested_)
      {
        fn_vect_.push_back(std::bind(
          autocontrast_caller,
          gpu_float_buffer_,
          input_fd.frame_res(),
          std::ref(compute_desc_),
          static_cast<cudaStream_t>(0)));

        autocontrast_requested_ = false;
        request_refresh();
      }

      if (compute_desc_.contrast_enabled)
      {
        fn_vect_.push_back(std::bind(
          manual_contrast_correction,
          gpu_float_buffer_,
          input_fd.frame_res(),
          65535,
          compute_desc_.contrast_min.load(),
          compute_desc_.contrast_max.load(),
          static_cast<cudaStream_t>(0)));
      }

      if (float_output_requested_)
      {
        fn_vect_.push_back(std::bind(
          &Pipe::record_float,
          this,
          gpu_float_buffer_));
      }

      if (autofocus_requested_)
      {
        fn_vect_.push_back(std::bind(
          &Pipe::autofocus_caller,
          this,
          gpu_float_buffer_,
          static_cast<cudaStream_t>(0)));
        autofocus_requested_ = false;
      }
    }

    fn_vect_.push_back(std::bind(
      float_to_ushort,
      gpu_float_buffer_,
      gpu_output_buffer_,
      input_fd.frame_res(),
      static_cast<cudaStream_t>(0)));
  }

  void Pipe::exec()
  {
    input_.flush();
    while (!termination_requested_)
    {
      if (input_.get_current_elts() >= input_length_)
      {
        for (FnType& f : fn_vect_) f();

        output_.enqueue(
          gpu_output_buffer_,
          cudaMemcpyDeviceToDevice);
        input_.dequeue();

        if (refresh_requested_)
          refresh();
      }
    }
  }
}