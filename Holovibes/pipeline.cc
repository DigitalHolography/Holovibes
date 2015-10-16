#include "pipeline.hh"

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
    , gpu_stft_buffer_(nullptr)
    , gpu_stft_dup_buffer_(nullptr)
    , gpu_float_buffer_(nullptr)
    , gpu_sqrt_vector_(nullptr)
    , gpu_lens_(nullptr)
    , plan3d_(0)
    , plan2d_(0)
    , plan1d_(0)
    , gpu_input_frame_ptr_(nullptr)
    , autofocus_requested_(false)
    , autocontrast_requested_(false)
    , stft_roi_requested_(false)
    , stft_roi_stop_requested_(false)
    , refresh_requested_(false)
    , update_n_requested_(false)
    , average_requested_(false)
    , average_record_requested_(false)
    , abort_construct_requested_(false)
    , average_output_(nullptr)
    , average_n_(0)
  {
    const unsigned short nsamples = desc.nsamples;

    /* if stft, we don't need to allocate more than one frame */
    if (compute_desc_.algorithm == ComputeDescriptor::STFT)
      input_length_ = 1;
    else
      input_length_ = nsamples;

    /* gpu_input_buffer */
    cudaMalloc<cufftComplex>(&gpu_input_buffer_,
      sizeof(cufftComplex)* input_.get_pixels() * input_length_);

    /* gpu_output_buffer */
    cudaMalloc<unsigned short>(&gpu_output_buffer_,
      sizeof(unsigned short)* input_.get_pixels());

    /* gpu_stft_buffer */
    cudaMalloc<cufftComplex>(&gpu_stft_buffer_,
      sizeof(cufftComplex)* compute_desc_.stft_roi_zone.load().area() * nsamples);

    /* gpu_stft_buffer */
    cudaMalloc<cufftComplex>(&gpu_stft_dup_buffer_,
      sizeof(cufftComplex)* compute_desc_.stft_roi_zone.load().area() * nsamples);

    /* gpu_float_buffer */
    cudaMalloc<float>(&gpu_float_buffer_,
      sizeof(float)* input_.get_pixels());

    /* Square root vector */
    cudaMalloc<float>(&gpu_sqrt_vector_, sizeof(float)* 65536);
    make_sqrt_vect(gpu_sqrt_vector_, 65535);

    /* gpu_lens */
    cudaMalloc(&gpu_lens_,
      input_.get_pixels() * sizeof(cufftComplex));

    /* CUFFT plan3d */
    cufftPlan3d(
      &plan3d_,
      input_length_,                   // NX
      input_.get_frame_desc().width,  // NY
      input_.get_frame_desc().height, // NZ
      CUFFT_C2C);

    /* CUFFT plan2d */
    cufftPlan2d(
      &plan2d_,
      input_.get_frame_desc().width,
      input_.get_frame_desc().height,
      CUFFT_C2C);

    /* CUFFT plan1d */
    cufftPlan1d(
      &plan1d_,
      nsamples,
      CUFFT_C2C,
      compute_desc_.stft_roi_zone.load().area() ? compute_desc_.stft_roi_zone.load().area() : 1
      );
    refresh();
  }

  Pipeline::~Pipeline()
  {
    /* CUFFT plan1d */
    cufftDestroy(plan1d_);

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

    /* gpu_stft_buffer */
    cudaFree(gpu_stft_buffer_);

    /* gpu_stft_dup_buffer */
    cudaFree(gpu_stft_dup_buffer_);
  }

  void Pipeline::update_n_parameter(unsigned short n)
  {
    unsigned int err_count = 0;
    abort_construct_requested_ = false;

    /* if stft, we don't need to allocate more than one frame */
    if (compute_desc_.algorithm == ComputeDescriptor::STFT)
      input_length_ = 1;
    else
      input_length_ = n;

    /*
    ** plan1D have limit and if they are reach, GPU may crash
    ** http://stackoverflow.com/questions/13187443/nvidia-cufft-limit-on-sizes-and-batches-for-fft-with-scikits-cuda
    ** 48e6 is an arbitary value
    */
    if (compute_desc_.stft_roi_zone.load().area() * static_cast<unsigned int>(n) > 48e6)
    {
      abort_construct_requested_ = true;
      std::cout
        << "[PREVENT_ERROR] pipeline l" << __LINE__ << " "
        << "You will reach the hard limit of cufftPlan\n"
        << compute_desc_.stft_roi_zone.load().area() * static_cast<unsigned int>(n)
        << " > "
        << 48e6
        << std::endl;
      return;
    }

    /* CUFFT plan3d realloc */
    if (plan3d_)
      cufftDestroy(plan3d_) ? ++err_count : 0;
    cufftPlan3d(
      &plan3d_,
      input_length_,                  // NX
      input_.get_frame_desc().width,  // NY
      input_.get_frame_desc().height, // NZ
      CUFFT_C2C) ? ++err_count : 0;

    /* CUFFT plan1d */
    if (plan1d_)
      cufftDestroy(plan1d_) ? ++err_count : 0;
    cufftPlan1d(
      &plan1d_,
      n,
      CUFFT_C2C,
      compute_desc_.stft_roi_zone.load().area() ? compute_desc_.stft_roi_zone.load().area() : 1
      ) ? ++err_count : 0;

    /* gpu_input_buffer */
    if (gpu_input_buffer_)
      cudaFree(gpu_input_buffer_) ? ++err_count : 0;
    gpu_input_buffer_ = nullptr;
    cudaMalloc<cufftComplex>(&gpu_input_buffer_,
      sizeof(cufftComplex) * input_.get_pixels() * input_length_) ? ++err_count : 0;

    /* gpu_stft_buffer */
    if (gpu_stft_buffer_)
      cudaFree(gpu_stft_buffer_) ? ++err_count : 0;
    gpu_stft_buffer_ = nullptr;
    cudaMalloc<cufftComplex>(&gpu_stft_buffer_,
      sizeof(cufftComplex) * compute_desc_.stft_roi_zone.load().area() * n) ? ++err_count : 0;

    /* gpu_stft_buffer */
    if (gpu_stft_dup_buffer_)
      cudaFree(gpu_stft_dup_buffer_) ? ++err_count : 0;
    gpu_stft_dup_buffer_ = nullptr;
    cudaMalloc<cufftComplex>(&gpu_stft_dup_buffer_,
      sizeof(cufftComplex) * compute_desc_.stft_roi_zone.load().area() * n) ? ++err_count : 0;
    if (err_count)
    {
      abort_construct_requested_ = true;
      std::cout
        << "[ERROR] pipeline l" << __LINE__
        << " err_count: " << err_count
        << " cudaError_t: " << cudaGetErrorString(cudaGetLastError())
        << std::endl;
    }
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

    if (abort_construct_requested_)
      return;

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
      input_length_,
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
    else if (compute_desc_.algorithm == ComputeDescriptor::STFT)
    {
      // Initialize FFT1 lens.
      fft1_lens(
        gpu_lens_,
        input_fd,
        compute_desc_.lambda,
        compute_desc_.zdistance);

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
        compute_desc_.pindex.load()));

      /* frame pointer */
      gpu_input_frame_ptr_ = gpu_input_buffer_;

      if (average_requested_)
      {
        fn_vect_.push_back(std::bind(
          &Pipeline::average_stft_caller,
          this,
          gpu_stft_dup_buffer_,
          compute_desc_.stft_roi_zone.load().get_width(),
          compute_desc_.stft_roi_zone.load().get_height(),
          compute_desc_.signal_zone.load(),
          compute_desc_.noise_zone.load(),
          compute_desc_.nsamples.load()));
        average_requested_ = false;
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
      if (average_record_requested_)
      {
        fn_vect_.push_back(std::bind(
          &Pipeline::average_record_caller,
          this,
          gpu_float_buffer_,
          input_fd.width,
          input_fd.height,
          compute_desc_.signal_zone.load(),
          compute_desc_.noise_zone.load()));

        average_record_requested_ = false;
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

    if (!float_output_requested_)
    {
      fn_vect_.push_back(std::bind(
        float_to_ushort,
        gpu_float_buffer_,
        gpu_output_buffer_,
        input_fd.frame_res()));
    }
    else
    {
      fn_vect_.push_back(std::bind(
        &Pipeline::record_float,
        this));
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

      if (!float_output_requested_)
      {
        output_.enqueue(
          gpu_output_buffer_,
          cudaMemcpyDeviceToDevice);
      }
      input_.dequeue();

      if (refresh_requested_)
        refresh();
    }
  }

  void Pipeline::request_refresh()
  {
    refresh_requested_ = true;
  }

  void Pipeline::request_float_output(std::string& float_output_file_src, unsigned int nb_frame)
  {
    try
    {
      float_output_file_.open(float_output_file_src, std::ofstream::trunc | std::ofstream::binary);
      float_output_nb_frame_ = nb_frame;
      float_output_requested_ = true;
      request_refresh();
      std::cout << "[PIPELINE]: float record start." << std::endl;
    }
    catch (std::exception& e)
    {
      std::cout << "[PIPELINE]: float record: " << e.what() << std::endl;
      request_float_output_stop();
    }
  }

  void Pipeline::request_float_output_stop()
  {
    if (float_output_file_.is_open())
      float_output_file_.close();
    float_output_requested_ = false;
    request_refresh();
    std::cout << "[PIPELINE]: float record done." << std::endl;
  }

  void Pipeline::record_float()
  {
    if (float_output_nb_frame_-- > 0)
    {
      unsigned int size = input_.get_pixels() * sizeof(float);
      // can be improve
      char *buf = new char[size];

      cudaMemcpy(buf, gpu_float_buffer_, size, cudaMemcpyDeviceToHost);
      float_output_file_.write(buf, size);

      // can be improve
      delete[] buf;
    }
    else
      request_float_output_stop();
  }

  void Pipeline::request_autocontrast()
  {
    autocontrast_requested_ = true;
    request_refresh();
  }

  void Pipeline::request_stft_roi()
  {
    request_update_n(compute_desc_.nsamples.load());
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
    average_record_requested_ = true;
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

  void Pipeline::average_stft_caller(
    cufftComplex*    input,
    unsigned int     width,
    unsigned int     height,
    Rectangle&       signal_zone,
    Rectangle&       noise_zone,
    unsigned int     nsamples)
  {
    unsigned int    i;
    cufftComplex*   cbuf;
    float*          fbuf;

    cudaMalloc<cufftComplex>(&cbuf, width * height * sizeof(cufftComplex));
    cudaMalloc<float>(&fbuf, width * height * sizeof(float));

    average_output_->resize(nsamples);
    for (i = 0; i < nsamples; ++i)
    {
      (*average_output_)[i] = (make_average_stft_plot(cbuf, fbuf, input, width, height, signal_zone, noise_zone, i, nsamples));
    }

    cudaFree(cbuf);
    cudaFree(fbuf);
  }
  /* Looks like the pipeline, but it searches for the right z value.
  The method choosen, iterates on the numbers of points given by the user
  between min and max, take the max and increase the precision to focus
  around the max more and more according to the number of iterations
  choosen.
  If the users chooses 2 iterations, a max will be choosen out of 10 images
  done between min and max. This max will be added and substracted a value
  in order to have a new (more accurate) zmin and zmax. And 10 more images
  will be produced, giving a better zmax
  */
  void Pipeline::autofocus_caller()
  {
    float z_min = compute_desc_.autofocus_z_min;
    float z_max = compute_desc_.autofocus_z_max;
    const unsigned int z_div = compute_desc_.autofocus_z_div;
    const unsigned int z_iter = compute_desc_.autofocus_z_iter;

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
    const size_t gpu_input_buffer_size = input_.get_pixels() * compute_desc_.nsamples * sizeof(cufftComplex);
    cudaMalloc(&gpu_input_buffer_tmp, gpu_input_buffer_size);
    float z_step = (z_max - z_min) / float(z_div);

    std::vector<float> focus_metric_values;

    /* Compute square af zone. */
    float* gpu_float_buffer_af_zone;
    const unsigned int zone_width = zone.top_right.x - zone.top_left.x;
    const unsigned int zone_height = zone.bottom_left.y - zone.top_left.y;

    const unsigned int af_square_size =
      static_cast<unsigned int>(powf(2, ceilf(log2f(zone_width > zone_height ? float(zone_width) : float(zone_height)))));
    const unsigned int af_size = af_square_size * af_square_size;

    cudaMalloc(&gpu_float_buffer_af_zone, af_size * sizeof(float));
    cudaMemset(gpu_float_buffer_af_zone, 0, af_size * sizeof(float));

    /// The main loop that calculates all z, and find the max one
    // z_step will decrease and zmin and zmax will merge into
    // the best autofocus_value

    float af_z = 0.0f;
    auto biggest = focus_metric_values.begin();
    unsigned int max_pos = 0;

    for (unsigned i = 0; i < z_iter; ++i)
    {
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

        const float focus_metric_value = focus_metric(gpu_float_buffer_af_zone, af_square_size);

        if (!std::isnan(focus_metric_value))
          focus_metric_values.push_back(focus_metric_value);
        else
          focus_metric_values.push_back(0);
      }
      /* Find max z */
      biggest = std::max_element(focus_metric_values.begin(), focus_metric_values.end());

      /* Case the max has not been found. */
      if (biggest == focus_metric_values.end())
        biggest = focus_metric_values.begin();
      max_pos = std::distance(focus_metric_values.begin(), biggest);

      // This is our temp max
      af_z = z_min + max_pos * z_step;

      // Calculation of the new max/min, taking the old step
      z_min = af_z - z_step;
      z_max = af_z + z_step;

      z_step = (z_max - z_min) / float(z_div);
      focus_metric_values.clear();
    }

    /// End of the loop, free resources and notify the new z

    compute_desc_.zdistance = af_z;
    compute_desc_.notify_observers();

    cudaFree(gpu_float_buffer_af_zone);
    cudaFree(gpu_input_buffer_tmp);
  }
}