#include <cassert>
#include <algorithm>

#include "pipe.hh"
#include "config.hh"
#include "info_manager.hh"
#include "compute_descriptor.hh"
#include "queue.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"

#include "fft1.cuh"
#include "fft2.cuh"
#include "filter2D.cuh"
#include "stft.cuh"
#include "convolution.cuh"
#include "flowgraphy.cuh"
#include "tools.cuh"
#include "autofocus.cuh"
#include "tools_conversion.cuh"
#include "tools.hh"
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
      sizeof (cufftComplex) * input_.get_pixels());

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
	cudaFree(gpu_input_buffer_);

	if (compute_desc_.stft_enabled)
		/*We malloc 2 frames because we might need a second one if the vibrometry is enabled*/
		cudaMalloc<cufftComplex>(&gpu_input_buffer_,
		sizeof(cufftComplex) * input_.get_pixels() * 2);
	else
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
	if (update_acc_requested_)
	{
		update_acc_requested_ = false;
		update_acc_parameter();
	}

	if (update_ref_diff_requested_)
	{
		update_ref_diff_requested_ = false;
		update_ref_diff_parameter();
		ref_diff_counter = compute_desc_.ref_diff_level.load();
	}

    if (abort_construct_requested_)
      return;

    if (autofocus_requested_)
    {
      fn_vect_.push_back(std::bind(
        &Pipe::autofocus_caller,
        this,
        gpu_float_buffer_,
        static_cast<cudaStream_t>(0)));
      autofocus_requested_ = false;
      request_refresh();
      return;
    }

    // Fill input complex buffer, one frame at a time.
	 fn_vect_.push_back(std::bind(
      make_contiguous_complex,
      std::ref(input_),
      gpu_input_buffer_,
      input_length_,
      static_cast<cudaStream_t>(0)));

	 unsigned int nframes = compute_desc_.nsamples.load();
	 unsigned int pframe = compute_desc_.pindex.load();
	 unsigned int qframe = compute_desc_.vibrometry_q.load();
	 if (compute_desc_.stft_enabled || (compute_desc_.filter_2d_enabled && !compute_desc_.stft_roi_zone.load().area()))
	 {
		 nframes = 1;
		 pframe = 0;
		 qframe = 0;
	 }

	 // In this case the q frame is not needed and will therefore not be computed in FFT1 & FFT2 
	 if (!compute_desc_.vibrometry_enabled)
		 qframe = pframe;
	 /* p frame pointer */
	 gpu_input_frame_ptr_ = gpu_input_buffer_ + pframe * input_fd.frame_res();

	 if (compute_desc_.ref_diff_enabled)
		 fn_vect_.push_back(std::bind(
		 &Pipe::handle_reference,
		 this,
		 gpu_input_buffer_,
		 nframes));

	 if (compute_desc_.ref_sliding_enabled)
		 fn_vect_.push_back(std::bind(
		 &Pipe::handle_sliding_reference,
		 this,
		 gpu_input_buffer_,
		 nframes));

	 if (compute_desc_.filter_2d_enabled)
	 {
		 fn_vect_.push_back(std::bind(
			 filter2D,
			 gpu_input_buffer_,
			 gpu_filter2d_buffer,
			 plan2d_,
			 compute_desc_.stft_roi_zone.load(),
			 input_fd,
			 static_cast<cudaStream_t>(0)));
	 }

	 if (!compute_desc_.filter_2d_enabled || (compute_desc_.filter_2d_enabled &&  compute_desc_.stft_roi_zone.load().area()))
	 {
		 if (compute_desc_.algorithm == ComputeDescriptor::None)
		 {
			 // Add temporal FFT1 1D .
			 fn_vect_.push_back(std::bind(
				 demodulation,
				 gpu_input_buffer_,
				 plan1d_,
				 static_cast<cudaStream_t>(0)));
		 }
		 else if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
		 {
			 fft1_lens(
				 gpu_lens_,
				 input_fd,
				 compute_desc_.lambda,
				 compute_desc_.zdistance,
				 static_cast<cudaStream_t>(0));
			 // Add FFT1.
			 fn_vect_.push_back(std::bind(
				 fft_1,
				 gpu_input_buffer_,
				 gpu_lens_,
				 plan1d_,
				 plan2d_,
				 input_fd.frame_res(),
				 nframes,
				 pframe,
				 qframe,
				 static_cast<cudaStream_t>(0)));

			 if (compute_desc_.vibrometry_enabled)
			 {

				 /* q frame pointer */
				 cufftComplex* q = gpu_input_buffer_ + qframe * input_fd.frame_res();

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
			 fft2_lens(
				 gpu_lens_,
				 input_fd,
				 compute_desc_.lambda,
				 compute_desc_.zdistance,
				 static_cast<cudaStream_t>(0));

			 fn_vect_.push_back(std::bind(
				 fft_2,
				 gpu_input_buffer_,
				 gpu_lens_,
				 plan1d_,
				 plan2d_,
				 input_fd.frame_res(),
				 nframes,
				 pframe,
				 qframe,
				 static_cast<cudaStream_t>(0)));

			 if (compute_desc_.vibrometry_enabled)
			 {
				 /* q frame pointer */
				 cufftComplex* q = gpu_input_buffer_ + qframe * input_fd.frame_res();

				 fn_vect_.push_back(std::bind(
					 frame_ratio,
					 gpu_input_frame_ptr_,
					 q,
					 gpu_input_frame_ptr_,
					 input_fd.frame_res(),
					 static_cast<cudaStream_t>(0)));
			 }
		 }
	 }
	 if (compute_desc_.stft_enabled)
	 {
		 fn_vect_.push_back(std::bind(
			 &ICompute::queue_enqueue,
			 this,
			 gpu_input_frame_ptr_,
			 gpu_stft_queue_));

		 fn_vect_.push_back(std::bind(
			 &ICompute::stft_handler,
			 this,
			 gpu_input_buffer_,
			 static_cast<cufftComplex *>(gpu_stft_queue_->get_buffer())));

		 if (compute_desc_.vibrometry_enabled)
		 {
			 qframe = 1;
			 /* q frame pointer */
			 cufftComplex* q = gpu_input_buffer_ + qframe * input_fd.frame_res();
			 fn_vect_.push_back(std::bind(
				 frame_ratio,
				 gpu_input_buffer_,
				 q,
				 gpu_input_buffer_,
				 input_fd.frame_res(),
				 static_cast<cudaStream_t>(0)));
		 }
		 /* frame pointer */
		 gpu_input_frame_ptr_ = gpu_input_buffer_;
	 }

	if (compute_desc_.convolution_enabled)
	{
		gpu_special_queue_start_index = 0;
		gpu_special_queue_max_index = compute_desc_.special_buffer_size.load();
		fn_vect_.push_back(std::bind(
			convolution_kernel,
			gpu_input_frame_ptr_,
			gpu_special_queue_,
			input_fd.frame_res(),
			input_fd.width,
			gpu_kernel_buffer_,
			compute_desc_.convo_matrix_width.load(),
			compute_desc_.convo_matrix_height.load(),
			compute_desc_.convo_matrix_z.load(),
			gpu_special_queue_start_index,
			gpu_special_queue_max_index,
			static_cast<cudaStream_t>(0)));
	}

	if (compute_desc_.flowgraphy_enabled)
	{
		gpu_special_queue_start_index = 0;
		gpu_special_queue_max_index = compute_desc_.special_buffer_size.load();
		fn_vect_.push_back(std::bind(
			convolution_flowgraphy,
			gpu_input_frame_ptr_,
			gpu_special_queue_,
			std::ref(gpu_special_queue_start_index),
			gpu_special_queue_max_index,
			input_fd.frame_res(),
			input_fd.width,
			compute_desc_.flowgraphy_level.load(),
			static_cast<cudaStream_t>(0)));
	}

	if (complex_output_requested_)
	{
		fn_vect_.push_back(std::bind(
			&Pipe::record_complex,
			this,
			gpu_input_frame_ptr_,
			static_cast<cudaStream_t>(0)));
	}
	
    /* Apply conversion to floating-point respresentation. */
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

		if (unwrap_2d_requested_)
		{
			if (!unwrap_res_2d_)
			{
				unwrap_res_2d_.reset(new UnwrappingResources_2d(
					input_.get_pixels()));
			}
			if (unwrap_res_2d_->image_resolution_ != input_.get_pixels())
				unwrap_res_2d_->reallocate(input_.get_pixels());

			fn_vect_.push_back(std::bind(
				unwrap_2d,
				gpu_float_buffer_,
				plan2d_,
				unwrap_res_2d_.get(),
				input_.get_frame_desc(),
				unwrap_res_2d_->gpu_angle_,
				static_cast<cudaStream_t>(0)));

			// Converting angle information in floating-point representation.
			fn_vect_.push_back(std::bind(
				rescale_float_unwrap2d,
				unwrap_res_2d_->gpu_angle_,
				gpu_float_buffer_,
				unwrap_res_2d_->minmax_buffer_,
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));
		}
		else
		{
			// Converting angle information in floating-point representation.
			fn_vect_.push_back(std::bind(
				rescale_argument,
				gpu_float_buffer_,
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));
		}
	}
	else if (compute_desc_.view_mode == ComputeDescriptor::COMPLEX)
	{
		fn_vect_.push_back(std::bind(
			complex_to_complex,
			gpu_input_frame_ptr_,
			gpu_output_buffer_,
			input_fd.frame_res() * 8,
			static_cast<cudaStream_t>(0)));
		return;
	}
    else
    {
		//Unwrap_res is ressources for phase_increase
		if (!unwrap_res_)
		{
			unwrap_res_.reset(new UnwrappingResources(
				compute_desc_.unwrap_history_size,
				input_.get_pixels()));
		}
		unwrap_res_->reset(compute_desc_.unwrap_history_size);
		unwrap_res_->reallocate(input_.get_pixels());
		if (compute_desc_.view_mode == holovibes::ComputeDescriptor::PHASE_INCREASE)
		{
			// Phase increase, complex multiply-with-conjugate method
			fn_vect_.push_back(std::bind(
				phase_increase,
				gpu_input_frame_ptr_,
				unwrap_res_.get(),
				input_fd.frame_res()));
		}
		else
		{
			// Fallback on modulus mode.
			fn_vect_.push_back(std::bind(
				complex_to_modulus,
				gpu_input_frame_ptr_,
				gpu_float_buffer_,
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));
		};

		if (unwrap_2d_requested_)
		{
			if (!unwrap_res_2d_)
			{
				unwrap_res_2d_.reset(new UnwrappingResources_2d(
					input_.get_pixels()));
			}
			if (unwrap_res_2d_->image_resolution_ != input_.get_pixels())
				unwrap_res_2d_->reallocate(input_.get_pixels());

			fn_vect_.push_back(std::bind(
				unwrap_2d,
				unwrap_res_->gpu_angle_current_,
				plan2d_,
				unwrap_res_2d_.get(),
				input_.get_frame_desc(),
				unwrap_res_2d_->gpu_angle_,
				static_cast<cudaStream_t>(0)));

			// Converting angle information in floating-point representation.
			fn_vect_.push_back(std::bind(
				rescale_float_unwrap2d,
				unwrap_res_2d_->gpu_angle_,
				gpu_float_buffer_,
				unwrap_res_2d_->minmax_buffer_,
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));
		}
		else
		{
			// Converting angle information in floating-point representation.
			fn_vect_.push_back(std::bind(
				rescale_float,
				unwrap_res_->gpu_angle_current_,
				gpu_float_buffer_,
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));
		}
    }


	/*Compute Accumulation buffer into gpu_float_buffer*/
	if (compute_desc_.img_acc_enabled)
	{
		/*Add image to phase accumulation buffer*/

		fn_vect_.push_back(std::bind(
			&ICompute::queue_enqueue,
			this,
			gpu_float_buffer_,
			gpu_img_acc_));

		fn_vect_.push_back(std::bind(
			accumulate_images,
			static_cast<float *>(gpu_img_acc_->get_buffer()),
			gpu_float_buffer_,
			gpu_img_acc_->get_start_index(),
			gpu_img_acc_->get_max_elts(),
			compute_desc_.img_acc_level.load(),
			input_fd.frame_res(),
			static_cast<cudaStream_t>(0)));
	}

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
        gpu_float_buffer_,
        static_cast<cudaStream_t>(0)));
    }

    if (gui::InfoManager::get_manager())
      fn_vect_.push_back(std::bind(
      &Pipe::fps_count,
      this));
	
    fn_vect_.push_back(std::bind(
      float_to_ushort,
      gpu_float_buffer_,
      gpu_output_buffer_,
      input_fd.frame_res(),
      static_cast<cudaStream_t>(0)));
  }

  void Pipe::autofocus_caller(float* input, cudaStream_t stream)
  {
    /* Fill gpu_input complex buffer. */
    make_contiguous_complex(
      input_,
      gpu_input_buffer_,
      compute_desc_.nsamples.load());

    float z_min = compute_desc_.autofocus_z_min;
    float z_max = compute_desc_.autofocus_z_max;
    const float z_div = static_cast<float>(compute_desc_.autofocus_z_div);
    Rectangle zone = compute_desc_.autofocus_zone;

    /* Autofocus needs to work on the same images.
    * It will computes on copies. */
    cufftComplex* gpu_input_buffer_tmp;
    const size_t gpu_input_buffer_size = input_.get_pixels() * compute_desc_.nsamples * sizeof(cufftComplex);
    cudaMalloc(&gpu_input_buffer_tmp, gpu_input_buffer_size);
    float z_step = (z_max - z_min) / z_div;

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
    // the best autofocus_value.
    float af_z = 0.0f;

    std::vector<float> focus_metric_values;
    auto biggest = focus_metric_values.begin();

    const camera::FrameDescriptor& input_fd = input_.get_frame_desc();

    unsigned int max_pos = 0;
    const unsigned int z_iter = compute_desc_.autofocus_z_iter;

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
			  plan1d_,
			  plan2d_,
			  input_fd.frame_res(),
			  compute_desc_.nsamples,
			  compute_desc_.pindex.load(),
			  compute_desc_.pindex.load());

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
            plan1d_,
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
        // TODO: Pop a window to warn the user
        else
        {
          cudaFree(gpu_float_buffer_af_zone);
          cudaFree(gpu_input_buffer_tmp);
          return;
        }

        if (compute_desc_.shift_corners_enabled)
        {
          shift_corners(
            gpu_float_buffer_,
            output_.get_frame_desc().width,
            output_.get_frame_desc().height);
        }

        if (compute_desc_.contrast_enabled)
        {
          manual_contrast_correction(
            gpu_float_buffer_,
            input_fd.frame_res(),
            65535,
            compute_desc_.contrast_min.load(),
            compute_desc_.contrast_max.load());
        }

        float_to_ushort(gpu_float_buffer_, gpu_output_buffer_, input_fd.frame_res());
        output_.enqueue(gpu_output_buffer_, cudaMemcpyDeviceToDevice);

        frame_memcpy(gpu_float_buffer_, zone, input_fd.width, gpu_float_buffer_af_zone, af_square_size);

        const float focus_metric_value = focus_metric(gpu_float_buffer_af_zone, af_square_size, stream, compute_desc_.autofocus_size);

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

      z_step = (z_max - z_min) / z_div;
      focus_metric_values.clear();
    }
	auto manager = gui::InfoManager::get_manager();
	manager->remove_info("Status");
    /// End of the loop, free resources and notify the new z
	// Sometimes a value outside the initial upper and lower bounds can be found
	// Thus checking if af_z is within initial bounds
	if (af_z != 0 && af_z >= compute_desc_.autofocus_z_min && af_z <= compute_desc_.autofocus_z_max)
		compute_desc_.zdistance = af_z;
    compute_desc_.notify_observers();

    cudaFree(gpu_float_buffer_af_zone);
    cudaFree(gpu_input_buffer_tmp);
  }


  void Pipe::exec()
  {
    if (global::global_config.flush_on_refresh)
      input_.flush();
    while (!termination_requested_)
    {
      if (input_.get_current_elts() >= input_length_)
      {
        for (FnType& f : fn_vect_) f();
		if (compute_desc_.view_mode != ComputeDescriptor::COMPLEX)
        output_.enqueue(
          gpu_output_buffer_,
          cudaMemcpyDeviceToDevice);
	else
		output_.enqueue(
			gpu_input_frame_ptr_,
			cudaMemcpyDeviceToDevice);
		input_.dequeue();
        if (refresh_requested_)
          refresh();
      }
    }
  }
}