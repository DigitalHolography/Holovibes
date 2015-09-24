#include "pipeline.hh"

#include <cassert>
#include <algorithm>

#include "fft1.cuh"
#include "fft2.cuh"
#include "tools.cuh"
#include "preprocessing.cuh"
#include "contrast_correction.cuh"
#include "vibrometry.cuh"
#include "average.cuh"
#include "autofocus.cuh"

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
    , average_requested_(false)
    , average_record_requested(false)
    , average_output_(nullptr)
    , average_n_(0)
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
    cudaMalloc<float>(&gpu_sqrt_vector_, sizeof(float) * 65536);
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
    /* Reset refresh flag. */
    refresh_requested_ = false;

    const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
    const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

    /* Clean current vector. */
    fn_vect_.clear();

    if (update_n_requested_)
    {
      update_n_requested_ = false;
      update_n_parameter(compute_desc_.nsamples);
    }

    if (autofocus_requested_)
    {
      fn_vect_.push_back(std::bind(
        &Pipeline::autofocus_caller,
        this));
      autofocus_requested_ = false;
      request_refresh();
      return;
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
          input_fd.frame_res()));
      }
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      fft2_lens(
        gpu_lens_,
        input_fd,
        compute_desc_.lambda,
        compute_desc_.zdistance);

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
          compute_desc_.vibrometry_q.load()));

        /* q frame pointer */
        cufftComplex* q = gpu_input_buffer_ + compute_desc_.vibrometry_q * input_fd.frame_res();

        fn_vect_.push_back(std::bind(
          frame_ratio,
          gpu_input_frame_ptr_,
          q,
          gpu_input_frame_ptr_,
          input_fd.frame_res()));
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
          compute_desc_.pindex.load()));
      }
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

    if (compute_desc_.shift_corners_enabled)
    {
      fn_vect_.push_back(std::bind(
        shift_corners,
        gpu_float_buffer_,
        output_fd.width,
        output_fd.height));
    }

    if (average_requested_)
    {
      if (average_record_requested)
      {
        fn_vect_.push_back(std::bind(
          &Pipeline::average_record_caller,
          this,
          gpu_float_buffer_,
          input_fd.width,
          input_fd.height,
          compute_desc_.signal_zone.load(),
          compute_desc_.noise_zone.load()));

        average_record_requested = false;
      }
      else
      {
        fn_vect_.push_back(std::bind(
          &Pipeline::average_caller,
          this,
          gpu_float_buffer_,
          input_fd.width,
          input_fd.height,
          compute_desc_.signal_zone.load(),
          compute_desc_.noise_zone.load()));
      }

      average_requested_ = false;
    }

    if (compute_desc_.log_scale_enabled)
    {
      fn_vect_.push_back(std::bind(
        apply_log10,
        gpu_float_buffer_,
        input_fd.frame_res()));
    }

    if (autocontrast_requested_)
    {
      fn_vect_.push_back(std::bind(
        autocontrast_caller,
        gpu_float_buffer_,
        input_fd.frame_res(),
        std::ref(compute_desc_)));

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
        compute_desc_.contrast_max.load()));
    }

    fn_vect_.push_back(std::bind(
      float_to_ushort,
      gpu_float_buffer_,
      gpu_output_buffer_,
      input_fd.frame_res()));
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
        refresh();
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
    autofocus_stop_requested_ = false;
    request_refresh();
  }

  void Pipeline::request_autofocus_stop()
  {
      autofocus_stop_requested_ = true;
  }

  void Pipeline::request_update_n(unsigned short n)
  {
    update_n_requested_ = true;
    compute_desc_.nsamples.exchange(n);
    request_refresh();
  }

  void Pipeline::request_average(
    ConcurrentDeque<std::tuple<float, float, float>>* output)
  {
    assert(output != nullptr);

    average_output_ = output;

    average_requested_ = true;
    request_refresh();
  }

  void Pipeline::request_average_record(
    ConcurrentDeque<std::tuple<float, float, float>>* output,
    unsigned int n)
  {
    assert(output != nullptr);
    assert(n != 0);

    average_output_ = output;
    average_n_ = n;

    average_requested_ = true;
    average_record_requested = true;
    request_refresh();
  }

  void Pipeline::autocontrast_caller(
    float* input,
    unsigned int size,
    ComputeDescriptor& compute_desc)
  {
    float min = 0.0f;
    float max = 0.0f;

    auto_contrast_correction(input, size, &min, &max);

    compute_desc.contrast_min = min;
    compute_desc.contrast_max = max;
    compute_desc.notify_observers();
  }

  void Pipeline::average_caller(
    float* input,
    unsigned int width,
    unsigned int height,
    Rectangle& signal,
    Rectangle& noise)
  {
    average_output_->push_back(make_average_plot(input, width, height, signal, noise));
  }

  void Pipeline::average_record_caller(
    float* input,
    unsigned int width,
    unsigned int height,
    Rectangle& signal,
    Rectangle& noise)
  {
    if (average_n_ > 0)
    {
      average_output_->push_back(make_average_plot(input, width, height, signal, noise));
      average_n_--;
    }
    else
    {
      average_n_ = 0;
      average_output_ = nullptr;
      request_refresh();
    }
  }

  /* Looks like the pipeline, but it search for the right z value. */
  void Pipeline::autofocus_caller()
  {
    float z_min = compute_desc_.autofocus_z_min;
    float z_max = compute_desc_.autofocus_z_max;
    unsigned int z_div = compute_desc_.autofocus_z_div;
    Rectangle zone = compute_desc_.autofocus_zone;

    const camera::FrameDescriptor& input_fd = input_.get_frame_desc();

    /* Fill gpu_input complex buffer. */

    make_contiguous_complex(
      input_,
      gpu_input_buffer_,
      compute_desc_.nsamples,
      gpu_sqrt_vector_);

    /* Autofocus needs to work on the same images.
     * It will computes on copies. */
    cufftComplex* gpu_input_buffer_tmp;
    size_t gpu_input_buffer_size = input_.get_pixels() * compute_desc_.nsamples * sizeof(cufftComplex);
    cudaMalloc(&gpu_input_buffer_tmp, gpu_input_buffer_size);
    float z_step = (z_max - z_min) / float(z_div);

    std::vector<float> focus_metric_values;

    /* Compute square af zone. */
    float* gpu_float_buffer_af_zone;
    unsigned int zone_width = zone.top_right.x - zone.top_left.x;
    unsigned int zone_height = zone.bottom_left.y - zone.top_left.y;

    unsigned int af_square_size =
      powf(2, ceilf(log2f(zone_width > zone_height ? float(zone_width) : float(zone_height))));
    unsigned int af_size = af_square_size * af_square_size;

    cudaMalloc(&gpu_float_buffer_af_zone, af_size * sizeof(float));
    cudaMemset(gpu_float_buffer_af_zone, 0, af_size * sizeof(float));

    for (float z = z_min; !autofocus_stop_requested_ && z < z_max; z += z_step)
    {
      /* Make input frames copies. */
      cudaMemcpy(
        gpu_input_buffer_tmp,
        gpu_input_buffer_,
        gpu_input_buffer_size,
        cudaMemcpyDeviceToDevice);

      if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
      {
        fft1_lens(
          gpu_lens_,
          input_fd,
          compute_desc_.lambda,
          z);

        fft_1(
          gpu_input_buffer_tmp,
          gpu_lens_,
          plan3d_,
          input_fd.frame_res(),
          compute_desc_.nsamples);

        gpu_input_frame_ptr_ = gpu_input_buffer_tmp + compute_desc_.pindex * input_fd.frame_res();
      }
      else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
      {
        fft2_lens(
          gpu_lens_,
          input_fd,
          compute_desc_.lambda,
          z);

        gpu_input_frame_ptr_ = gpu_input_buffer_tmp + compute_desc_.pindex * input_fd.frame_res();

        fft_2(
          gpu_input_buffer_tmp,
          gpu_lens_,
          plan3d_,
          plan2d_,
          input_fd.frame_res(),
          compute_desc_.nsamples,
          compute_desc_.pindex,
          compute_desc_.pindex);
      }
      else
        assert(!"Impossible case");

      if (compute_desc_.view_mode == ComputeDescriptor::MODULUS)
      {
        complex_to_modulus(gpu_input_frame_ptr_, gpu_float_buffer_, input_fd.frame_res());
      }
      else if (compute_desc_.view_mode == ComputeDescriptor::SQUARED_MODULUS)
      {
        complex_to_squared_modulus(gpu_input_frame_ptr_, gpu_float_buffer_, input_fd.frame_res());
      }
      else if (compute_desc_.view_mode == ComputeDescriptor::ARGUMENT)
      {
        complex_to_argument(gpu_input_frame_ptr_, gpu_float_buffer_, input_fd.frame_res());
      }
      else
        assert(!"Impossible case");

      if (compute_desc_.shift_corners_enabled)
      {
        shift_corners(
          gpu_float_buffer_,
          output_.get_frame_desc().width,
          output_.get_frame_desc().height);
      }

      float_to_ushort(gpu_float_buffer_, gpu_output_buffer_, input_fd.frame_res());
      output_.enqueue(gpu_output_buffer_, cudaMemcpyDeviceToDevice);

      frame_memcpy(gpu_float_buffer_, zone, input_fd.width, gpu_float_buffer_af_zone, af_square_size);

      float focus_metric_value = focus_metric(gpu_float_buffer_af_zone, af_square_size);

      if (!std::isnan(focus_metric_value))
        focus_metric_values.push_back(focus_metric_value);
      else
        focus_metric_values.push_back(0);
    }

    /* Find max z */
    auto biggest = std::max_element(focus_metric_values.begin(), focus_metric_values.end());
    /* Case the max has not been found. */
    if (biggest == focus_metric_values.end())
      biggest = focus_metric_values.begin();
    auto max_pos = std::distance(focus_metric_values.begin(), biggest);
    float af_z = z_min + max_pos * z_step;


    compute_desc_.zdistance = af_z;
    compute_desc_.notify_observers();

    cudaFree(gpu_float_buffer_af_zone);
    cudaFree(gpu_input_buffer_tmp);
  }
}
