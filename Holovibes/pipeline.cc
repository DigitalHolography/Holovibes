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
    , step_count_before_refresh_(0)
  {
    float                         *gpu_float_buffer = nullptr;

    /* The 16-bit buffer is allocated (and deallocated) separately,
     as no Module is associated directly to it. */
    cudaMalloc<unsigned short>(&gpu_short_buffer_,
      sizeof(unsigned short)* input_.get_pixels());

    cudaMalloc(&gpu_float_buffer, sizeof(float)* input_.get_pixels());
    gpu_float_buffers_.push_back(gpu_float_buffer);
    cudaMalloc(&gpu_float_buffer, sizeof(float)* input_.get_pixels());
    gpu_float_buffers_.push_back(gpu_float_buffer);

    update_n_parameter(compute_desc_.nsamples);
    refresh();
  }

  Pipeline::~Pipeline()
  {
    stop_pipeline();
    cudaFree(gpu_short_buffer_);
    delete_them(gpu_complex_buffers_, [](cufftComplex* buffer) { cudaFree(buffer); });
    delete_them(gpu_float_buffers_, [](float* buffer) { cudaFree(buffer); });
  }

  void Pipeline::stop_pipeline()
  {
    delete_them(modules_, [](Module* module) { delete module; });
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

        if (refresh_requested_
          && (step_count_before_refresh_ == 0 || --step_count_before_refresh_ == 0))
        {
          refresh();
        }
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

  void Pipeline::update_n_parameter(unsigned short n)
  {
    ICompute::update_n_parameter(n);

    cufftComplex                  *gpu_complex_buffer = nullptr;

    delete_them(gpu_complex_buffers_, [](cufftComplex* buffer) { cudaFree(buffer); });
    cudaMalloc(&gpu_complex_buffer, sizeof(cufftComplex)* input_.get_pixels() * input_length_);
    gpu_complex_buffers_.push_back(gpu_complex_buffer);
    cudaMalloc(&gpu_complex_buffer, sizeof(cufftComplex)* input_.get_pixels() * input_length_);
    gpu_complex_buffers_.push_back(gpu_complex_buffer);

    if (gpu_pindex_buffers_.size())
      gpu_pindex_buffers_.clear();
    std::for_each(gpu_complex_buffers_.begin(),
      gpu_complex_buffers_.end(),
      [&](cufftComplex* buf) { gpu_pindex_buffers_.push_back(buf + compute_desc_.pindex * input_.get_frame_desc().frame_res()); });
  }

  void Pipeline::refresh()
  {
    const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
    const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

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

    if (autofocus_requested_)
    {
      // LOL ...
    }

    modules_[0]->push_back_worker(std::bind(
      make_contiguous_complex,
      std::ref(input_),
      std::ref(gpu_complex_buffers_[0]),
      input_length_,
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
      modules_[1]->push_back_worker(std::bind(
        fft_1,
        std::ref(gpu_complex_buffers_[1]),
        gpu_lens_,
        plan3d_,
        input_fd.frame_res(),
        compute_desc_.nsamples.load(),
        modules_[1]->stream_
        ));

      if (compute_desc_.vibrometry_enabled)
      {
        /*
        // q frame pointer
        cufftComplex* q = gpu_input_buffer_ + compute_desc_.vibrometry_q * input_fd.frame_res();

        fn_vect_.push_back(std::bind(
        frame_ratio,
        gpu_input_frame_ptr_,
        q,
        gpu_input_frame_ptr_,
        input_fd.frame_res(),
        static_cast<cudaStream_t>(0)));
        */
      }
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      fft2_lens(
        gpu_lens_,
        input_fd,
        compute_desc_.lambda,
        compute_desc_.zdistance);

      if (compute_desc_.vibrometry_enabled)
      {
        /*
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

        // q frame pointer
        cufftComplex* q = gpu_input_buffer_ + compute_desc_.vibrometry_q * input_fd.frame_res();

        fn_vect_.push_back(std::bind(
        frame_ratio,
        gpu_input_frame_ptr_,
        q,
        gpu_input_frame_ptr_,
        input_fd.frame_res(),
        static_cast<cudaStream_t>(0)));
        */
      }
      else
      {
        modules_[1]->push_back_worker(std::bind(
          fft_2,
          std::ref(gpu_complex_buffers_[1]),
          gpu_lens_,
          plan3d_,
          plan2d_,
          input_fd.frame_res(),
          compute_desc_.nsamples.load(),
          compute_desc_.pindex.load(),
          compute_desc_.pindex.load(),
          modules_[1]->stream_
          ));
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
      modules_[1]->push_back_worker(std::bind(
        stft,
        std::ref(gpu_complex_buffers_[1]),
        gpu_lens_,
        gpu_stft_buffer_,
        gpu_stft_dup_buffer_,
        plan2d_,
        plan1d_,
        compute_desc_.stft_roi_zone.load(),
        curr_elt_stft_,
        input_fd,
        compute_desc_.nsamples.load(),
        modules_[1]->stream_
        ));

      modules_[1]->push_back_worker(std::bind(
        stft_recontruct,
        std::ref(gpu_complex_buffers_[1]),
        gpu_stft_dup_buffer_,
        compute_desc_.stft_roi_zone.load(),
        input_fd,
        (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_width() : input_fd.width),
        (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_height() : input_fd.height),
        compute_desc_.pindex.load(),
        compute_desc_.nsamples.load(),
        modules_[1]->stream_
        ));

      gpu_pindex_buffers_ = gpu_complex_buffers_;

      if (average_requested_)
      {
        if (compute_desc_.stft_roi_zone.load().area())
          modules_[1]->push_back_worker(std::bind(
          &Pipeline::average_stft_caller,
          this,
          gpu_stft_dup_buffer_,
          input_fd.width,
          input_fd.height,
          compute_desc_.stft_roi_zone.load().get_width(),
          compute_desc_.stft_roi_zone.load().get_height(),
          compute_desc_.signal_zone.load(),
          compute_desc_.noise_zone.load(),
          compute_desc_.nsamples.load(),
          modules_[1]->stream_
          ));
        average_requested_ = false;
      }
    }
    else
      assert(!"Impossible case.");

    /* Apply conversion to unsigned short. */
    if (compute_desc_.view_mode == ComputeDescriptor::MODULUS)
    {
      modules_[0]->push_front_worker(std::bind(
        complex_to_modulus,
        std::ref(gpu_pindex_buffers_[0]),
        std::ref(gpu_float_buffers_[0]),
        input_fd.frame_res(),
        modules_[0]->stream_
        ));
    }
    else if (compute_desc_.view_mode == ComputeDescriptor::SQUARED_MODULUS)
    {
      modules_[0]->push_front_worker(std::bind(
        complex_to_squared_modulus,
        std::ref(gpu_pindex_buffers_[0]),
        std::ref(gpu_float_buffers_[0]),
        input_fd.frame_res(),
        modules_[0]->stream_
        ));
    }
    else if (compute_desc_.view_mode == ComputeDescriptor::ARGUMENT)
    {
      modules_[0]->push_front_worker(std::bind(
        complex_to_argument,
        std::ref(gpu_pindex_buffers_[0]),
        std::ref(gpu_float_buffers_[0]),
        input_fd.frame_res(),
        modules_[0]->stream_
        ));
    }
    else
      assert(!"Impossible case.");

    /* [POSTPROCESSING] Everything behind this line uses output_frame_ptr */

    if (compute_desc_.shift_corners_enabled)
    {
      modules_[2]->push_back_worker(std::bind(
        shift_corners,
        std::ref(gpu_float_buffers_[1]),
        output_fd.width,
        output_fd.height,
        modules_[2]->stream_
        ));
    }

    if (average_requested_)
    {
      if (average_record_requested_)
      {
        modules_[2]->push_back_worker(std::bind(
          &Pipeline::average_record_caller,
          this,
          std::ref(gpu_float_buffers_[1]),
          input_fd.width,
          input_fd.height,
          compute_desc_.signal_zone.load(),
          compute_desc_.noise_zone.load(),
          modules_[2]->stream_
          ));

        average_record_requested_ = false;
      }
      else
      {
        modules_[2]->push_back_worker(std::bind(
          &Pipeline::average_caller,
          this,
          std::ref(gpu_float_buffers_[1]),
          input_fd.width,
          input_fd.height,
          compute_desc_.signal_zone.load(),
          compute_desc_.noise_zone.load(),
          modules_[2]->stream_
          ));
      }
      average_requested_ = false;
    }

    if (compute_desc_.log_scale_enabled)
    {
      modules_[2]->push_back_worker(std::bind(
        apply_log10,
        std::ref(gpu_float_buffers_[1]),
        input_fd.frame_res(),
        modules_[2]->stream_
        ));
    }

    if (autocontrast_requested_)
    {
      modules_[2]->push_back_worker(std::bind(
        autocontrast_caller,
        std::ref(gpu_float_buffers_[1]),
        input_fd.frame_res(),
        std::ref(compute_desc_),
        modules_[2]->stream_
        ));

      step_count_before_refresh_ = modules_.size() + 1;
      request_refresh();
      autocontrast_requested_ = false;
    }

    if (compute_desc_.contrast_enabled)
    {
      modules_[2]->push_back_worker(std::bind(
        manual_contrast_correction,
        std::ref(gpu_float_buffers_[1]),
        input_fd.frame_res(),
        65535,
        compute_desc_.contrast_min.load(),
        compute_desc_.contrast_max.load(),
        modules_[2]->stream_
        ));
    }

    if (!float_output_requested_)
    {
      modules_[2]->push_back_worker(std::bind(
        float_to_ushort,
        std::ref(gpu_float_buffers_[1]),
        gpu_short_buffer_,
        input_fd.frame_res(),
        modules_[2]->stream_
        ));
    }
    else
    {
      /*
      modules_[2]->push_back_worker(std::bind(
      &Pipe::record_float,
      this));
      */
    }
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