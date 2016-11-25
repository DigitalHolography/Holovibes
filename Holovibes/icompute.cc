#include <cufft.h>

#include "icompute.hh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "stft.cuh"
#include "tools.cuh"
#include "contrast_correction.cuh"
#include "preprocessing.cuh"
#include "autofocus.cuh"
#include "average.cuh"
#include "queue.hh"
#include "concurrent_deque.hh"
#include "compute_descriptor.hh"
#include "power_of_two.hh"
#include "info_manager.hh"
# include "compute_bundles.hh"


namespace holovibes
{
  ICompute::ICompute(
    Queue& input,
    Queue& output,
    ComputeDescriptor& desc)
    : compute_desc_(desc)
    , input_(input)
    , output_(output)
    , gpu_sqrt_vector_(nullptr)
    , unwrap_res_(nullptr)
    , gpu_stft_buffer_(nullptr)
    , gpu_lens_(nullptr)
	, gpu_kernel_buffer_(nullptr)
	, gpu_special_queue_(nullptr)
	, gpu_stft_queue_(nullptr)
	, gpu_ref_diff_queue(nullptr)
    , plan3d_(0)
    , plan2d_(0)
    , plan1d_(0)
	, plan1d_stft_(0)
    , unwrap_requested_(true)
    , autofocus_requested_(false)
    , autocontrast_requested_(false)
    , refresh_requested_(false)
    , update_n_requested_(false)
    , stft_update_roi_requested_(false)
    , average_requested_(false)
    , average_record_requested_(false)
    , abort_construct_requested_(false)
    , termination_requested_(false)
	, update_acc_requested_(false)
	, update_ref_diff_requested_(false)
    , q_gpu_stft_buffer_(nullptr)
    , average_output_(nullptr)
    , average_n_(0)
    , af_env_({ 0 })
    , past_time_(std::chrono::high_resolution_clock::now())
  {
    const unsigned short nsamples = desc.nsamples;

    /* if stft, we don't need to allocate more than one frame */
   // if (compute_desc_.stft_enabled)
  //    input_length_ = 1;
   // else
      input_length_ = nsamples;

	if (compute_desc_.stft_enabled) {
		cudaMalloc<cufftComplex>(&gpu_stft_buffer_,
			sizeof(cufftComplex)* input_.get_pixels() * nsamples);
	}
    /* Square root vector */
	/* TODO: not used anymore because square root of 65535 is not initialized */
	/* need to benchmark with & without this vector to know if it was useful */
   // cudaMalloc<float>(&gpu_sqrt_vector_, sizeof(float) * 65536);
    //make_sqrt_vect(gpu_sqrt_vector_, 65535);

    /* gpu_lens */
    cudaMalloc(&gpu_lens_,
      input_.get_pixels() * sizeof(cufftComplex));

    /* CUFFT plan3d */
    if (compute_desc_.algorithm == ComputeDescriptor::FFT1
      || compute_desc_.algorithm == ComputeDescriptor::FFT2)
      cufftPlan3d(
      &plan3d_,
      input_length_,                  // NX
      input_.get_frame_desc().width,  // NY
      input_.get_frame_desc().height, // NZ
      CUFFT_C2C);

    /* CUFFT plan2d */
    cufftPlan2d(
      &plan2d_,
      input_.get_frame_desc().width,
      input_.get_frame_desc().height,
      CUFFT_C2C);

	/* CUFFT plan1d temporal*/
	int inembed[1] = { input_length_ };

	cufftPlanMany(&plan1d_, 1, inembed,
		inembed, input_.get_pixels(), 1,
		inembed, input_.get_pixels(), 1,
		CUFFT_C2C, input_.get_pixels());

	if (compute_desc_.convolution_enabled
		|| compute_desc_.flowgraphy_enabled)
	{
		/* gpu_tmp_input */
		cudaMalloc<cufftComplex>(&gpu_tmp_input_,
			sizeof(cufftComplex)* input_.get_pixels() * compute_desc_.nsamples);
	}
	if (compute_desc_.convolution_enabled)
	{
		/* kst_size */
		int size = compute_desc_.convo_matrix.size();
		/* gpu_kernel_buffer */
		cudaMalloc<float>(&gpu_kernel_buffer_,
			sizeof (float)* (size));
		/* Build the kst 3x3 matrix */
		float* kst_complex_cpu = (float *) malloc(sizeof (float) * size);
		for (int i = 0; i < size; ++i)
		{
			kst_complex_cpu[i] = compute_desc_.convo_matrix[i];
			//kst_complex_cpu[i].y = 0;
		}
		cudaMemcpy(gpu_kernel_buffer_, kst_complex_cpu, sizeof (float) * size, cudaMemcpyHostToDevice);
	}
	if (compute_desc_.flowgraphy_enabled || compute_desc_.convolution_enabled)
	{
		/* gpu_tmp_input */
		cudaMalloc<cufftComplex>(&gpu_special_queue_,
			sizeof(cufftComplex) * input_.get_pixels() * compute_desc_.special_buffer_size.load());
	}

	if (compute_desc_.img_acc_enabled)
	{
		camera::FrameDescriptor new_fd = input_.get_frame_desc();
		new_fd.depth = 4;
		gpu_img_acc_ = new holovibes::Queue(new_fd, compute_desc_.img_acc_level.load(), "AccumulationQueue");
	}

	if (compute_desc_.stft_enabled)
	{
		cufftPlanMany(&plan1d_stft_, 1, inembed,
			inembed, input_.get_pixels(), 1,
			inembed, input_.get_pixels(), 1,
			CUFFT_C2C, input_.get_pixels());

		camera::FrameDescriptor new_fd2 = input_.get_frame_desc();
		new_fd2.depth = 8;
		gpu_stft_queue_ = new holovibes::Queue(new_fd2, compute_desc_.stft_level.load(), "STFTQueue");
	}

	if (compute_desc_.ref_diff_enabled)
	{
		gpu_ref_diff_queue = new holovibes::Queue(input_.get_frame_desc(), compute_desc_.stft_level.load(), "TakeRefQueue");
	}
  }

  ICompute::~ICompute()
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

    /* gpu_stft_buffer */
    cudaFree(gpu_stft_buffer_);

	/* gpu_special_queue */
	cudaFree(gpu_special_queue_);

    /* gpu_float_buffer_af_zone */
    cudaFree(af_env_.gpu_float_buffer_af_zone);

    /* gpu_input_buffer_tmp */
    cudaFree(af_env_.gpu_input_buffer_tmp);

    if (gui::InfoManager::get_manager())
      gui::InfoManager::get_manager()->remove_info("Rendering");

	/* gpu_kernel_buffer */
	cudaFree(gpu_kernel_buffer_);

	/* gpu_img_acc */
	delete gpu_img_acc_;

	/* gpu_stft_queue */
	delete gpu_stft_queue_;

	/* gpu_take_ref_queue */
	delete gpu_ref_diff_queue;
  }

  void ICompute::update_n_parameter(unsigned short n)
  {
    unsigned int err_count = 0;
    abort_construct_requested_ = false;

    /* if stft, we don't need to allocate more than one frame */
	if (!compute_desc_.stft_enabled)
		input_length_ = n;
	else
		input_length_ = 1;


    /* CUFFT plan3d realloc */
    cudaDestroy<cufftResult>(&plan3d_) ? ++err_count : 0;

    if (compute_desc_.algorithm == ComputeDescriptor::FFT1
      || compute_desc_.algorithm == ComputeDescriptor::FFT2)
      cufftPlan3d(
      &plan3d_,
      input_length_,                  // NX
      input_.get_frame_desc().width,  // NY
      input_.get_frame_desc().height, // NZ
      CUFFT_C2C) ? ++err_count : 0;

    /* CUFFT plan1d realloc */
    cudaDestroy<cufftResult>(&plan1d_) ? ++err_count : 0;

    /* gpu_stft_buffer */
    cudaDestroy<cudaError_t>(&gpu_stft_buffer_) ? ++err_count : 0;
	
	//int inembed[1] = { input_length_ };

	int inembed[1] = { input_length_ };

	cufftPlanMany(&plan1d_, 1, inembed,
		inembed, input_.get_pixels(), 1,
		inembed, input_.get_pixels(), 1,
		CUFFT_C2C, input_.get_pixels());
	
 if (compute_desc_.stft_enabled)
    {

	 /* CUFFT plan1d realloc */
	 cudaDestroy<cufftResult>(&plan1d_stft_);

	 int inembed_stft[1] = { n };

	 cufftPlanMany(&plan1d_stft_, 1, inembed_stft,
		 inembed_stft, input_.get_pixels(), 1,
		 inembed_stft, input_.get_pixels(), 1,
		 CUFFT_C2C, input_.get_pixels());
      /* gpu_stft_buffer */
     /* cudaMalloc(&gpu_stft_buffer_,
        sizeof(cufftComplex) * compute_desc_.stft_roi_zone.load().area() * n) ? ++err_count : 0;*/

	 cudaMalloc(&gpu_stft_buffer_,
		 sizeof(cufftComplex)* input_.get_pixels() * n) ? ++err_count : 0;

      /* gpu_stft_buffer */
     /* cudaMalloc(&gpu_stft_dup_buffer_,
        sizeof(cufftComplex)* compute_desc_.stft_roi_zone.load().area() * n) ? ++err_count : 0;
		*/
    }

 if (gpu_stft_queue_ != nullptr)
 {
	 delete gpu_stft_queue_;
	 gpu_stft_queue_ = nullptr;
 }

 if (compute_desc_.stft_enabled)
 {

	 camera::FrameDescriptor new_fd = input_.get_frame_desc();
	 new_fd.depth = 8;
	 gpu_stft_queue_ = new holovibes::Queue(new_fd, n, "STFTQueue");
 }


    if (err_count)
    {
      abort_construct_requested_ = true;
      std::cout
        << "[ERROR] ICompute l" << __LINE__
        << " err_count: " << err_count
        << " cudaError_t: " << cudaGetErrorString(cudaGetLastError())
        << std::endl;
    }
  }

  void ICompute::refresh()
  {
    if (compute_desc_.stft_enabled
      && compute_desc_.vibrometry_enabled)
    {
      cudaMalloc<cufftComplex>(&q_gpu_stft_buffer_,
        sizeof(cufftComplex)* input_.get_pixels());
    }
    else
      cudaDestroy<cudaError_t>(&q_gpu_stft_buffer_);

    if (!float_output_requested_ && !complex_output_requested_ && fqueue_)
    {
      delete fqueue_;
      fqueue_ = nullptr;
    }

	if (compute_desc_.convolution_enabled
		|| compute_desc_.flowgraphy_enabled)
	{
		/* gpu_tmp_input */
		cudaFree(gpu_tmp_input_);
		/* gpu_tmp_input */
		cudaMalloc<cufftComplex>(&gpu_tmp_input_,
			sizeof(cufftComplex)* input_.get_pixels() * compute_desc_.nsamples);
	}
	if (compute_desc_.convolution_enabled)
	{
		/* kst_size */
		int size = compute_desc_.convo_matrix.size();
		/* gpu_kernel_buffer */
		cudaFree(gpu_kernel_buffer_);
		/* gpu_kernel_buffer */
		cudaMalloc<float>(&gpu_kernel_buffer_,
			sizeof (float) * (size));
		/* Build the kst 3x3 matrix */
		float* kst_complex_cpu = (float *) malloc(sizeof (float) * size);
		for (int i = 0; i < size; ++i)
		{
			kst_complex_cpu[i] = compute_desc_.convo_matrix[i];
			//kst_complex_cpu[i].y = 0;
		}
		cudaMemcpy(gpu_kernel_buffer_, kst_complex_cpu, sizeof (float) * size, cudaMemcpyHostToDevice);
	}
	/* not deleted properly !!!!*/
	if (compute_desc_.flowgraphy_enabled || compute_desc_.convolution_enabled)
	{
		/* gpu_tmp_input */
		cudaFree(gpu_special_queue_);
		/* gpu_tmp_input */
		cudaMalloc<cufftComplex>(&gpu_special_queue_,
			sizeof(cufftComplex)* input_.get_pixels() * compute_desc_.special_buffer_size.load());
	}

  }

  void ICompute::update_acc_parameter()
  {
	  if (gpu_img_acc_ != nullptr)
	  {
		  delete gpu_img_acc_;
		  gpu_img_acc_ = nullptr;
	  }
	  if (compute_desc_.img_acc_enabled)
	  {
		  camera::FrameDescriptor new_fd = input_.get_frame_desc();
		  new_fd.depth = 4;
		  gpu_img_acc_ = new holovibes::Queue(new_fd, compute_desc_.img_acc_level.load(), "Accumulation");
	  }
  }

  void ICompute::update_ref_diff_parameter()
  {
	  if (gpu_ref_diff_queue != nullptr)
	  {
		  delete gpu_ref_diff_queue;
		  gpu_ref_diff_queue = nullptr;
	  }
	  if (compute_desc_.ref_diff_enabled)
	  {
		  gpu_ref_diff_queue = new holovibes::Queue(input_.get_frame_desc(), compute_desc_.ref_diff_level, "TakeRefQueue");
	  }
  }

  void ICompute::request_refresh()
  {
    refresh_requested_ = true;
  }

  void ICompute::request_acc_refresh()
  {
	  update_acc_requested_ = true;
	  request_refresh();
  }

  void ICompute::request_ref_diff_refresh()
  {
	  update_ref_diff_requested_ = true;
	  request_refresh();
  }

  void ICompute::request_float_output(Queue* fqueue)
  {
    fqueue_ = fqueue;
    float_output_requested_ = true;
    request_refresh();
  }

  void ICompute::request_float_output_stop()
  {
    float_output_requested_ = false;
    request_refresh();
  }

  void ICompute::request_complex_output(Queue* fqueue)
  {
	  fqueue_ = fqueue;
	  complex_output_requested_ = true;
	  request_refresh();
  }

  void ICompute::request_complex_output_stop()
  {
	  complex_output_requested_ = false;
	  request_refresh();
  }


  void ICompute::request_termination()
  {
    termination_requested_ = true;
  }

  void ICompute::request_autocontrast()
  {
    autocontrast_requested_ = true;
    request_refresh();
  }

  void ICompute::request_stft_roi_update()
  {
    stft_update_roi_requested_ = true;
    request_update_n(compute_desc_.nsamples.load());
  }

  void ICompute::request_stft_roi_end()
  {
    stft_update_roi_requested_ = false;
    request_update_n(compute_desc_.nsamples.load());
  }

  void ICompute::request_autofocus()
  {
    autofocus_requested_ = true;
    autofocus_stop_requested_ = false;
    request_refresh();
  }

  void ICompute::request_autofocus_stop()
  {
    autofocus_stop_requested_ = true;
  }

  void ICompute::request_update_n(const unsigned short n)
  {
    update_n_requested_ = true;
    compute_desc_.nsamples.exchange(n);
    request_refresh();
  }

  void ICompute::request_update_unwrap_size(const unsigned size)
  {
    compute_desc_.unwrap_history_size.exchange(size);
    request_refresh();
  }

  void ICompute::request_unwrapping(const bool value)
  {
    unwrap_requested_ = value;
  }

  void ICompute::request_average(
    ConcurrentDeque<std::tuple<float, float, float, float>>* output)
  {
    assert(output != nullptr);

    if (compute_desc_.stft_enabled)
      output->resize(compute_desc_.nsamples.load());
    average_output_ = output;

    average_requested_ = true;
    request_refresh();
  }
  void ICompute::request_average_stop()
  {
    average_requested_ = false;
    request_refresh();
  }

  void ICompute::request_average_record(
    ConcurrentDeque<std::tuple<float, float, float, float>>* output,
    const unsigned int n)
  {
    assert(output != nullptr);
    assert(n != 0);

    average_output_ = output;
    average_n_ = n;

    average_requested_ = true;
    average_record_requested_ = true;
    request_refresh();
  }

  void ICompute::autocontrast_caller(
    float* input,
    const unsigned int size,
    ComputeDescriptor& compute_desc,
    cudaStream_t stream)
  {
    float min = 0.0f;
    float max = 0.0f;

    auto_contrast_correction(input, size, &min, &max, stream);

    compute_desc.contrast_min = min;
    compute_desc.contrast_max = max;
    compute_desc.notify_observers();
  }

  void ICompute::record_float(float* float_output, cudaStream_t stream)
  {
    // TODO: use stream in enqueue
    fqueue_->enqueue(float_output, cudaMemcpyDeviceToDevice);
  }

  void ICompute::record_complex(cufftComplex* complex_output, cudaStream_t stream)
  {
	  // TODO: use stream in enqueue aswell
	  fqueue_->enqueue(complex_output, cudaMemcpyDeviceToDevice);
  }

  void ICompute::queue_enqueue(void* input, Queue* queue)
  {
	  queue->enqueue(input, cudaMemcpyDeviceToDevice, true);
  }

  void ICompute::average_caller(
    float* input,
    const unsigned int width,
    const unsigned int height,
    const Rectangle& signal,
    const Rectangle& noise,
    cudaStream_t stream)
  {
    average_output_->push_back(make_average_plot(input, width, height, signal, noise, stream));
  }

  void ICompute::average_record_caller(
    float* input,
    const unsigned int width,
    const unsigned int height,
    const Rectangle& signal,
    const Rectangle& noise,
    cudaStream_t stream)
  {
    if (average_n_ > 0)
    {
      average_output_->push_back(make_average_plot(input, width, height, signal, noise, stream));
      average_n_--;
    }
    else
    {
      average_n_ = 0;
      average_output_ = nullptr;
      request_refresh();
    }
  }

  void ICompute::average_stft_caller(
    cufftComplex* stft_buffer,
    const unsigned int width,
    const unsigned int height,
    const unsigned int width_roi,
    const unsigned int height_roi,
    Rectangle& signal_zone,
    Rectangle& noise_zone,
    const unsigned int nsamples,
    cudaStream_t stream)
  {
    cufftComplex*   cbuf;
    float*          fbuf;

    if (cudaMalloc<cufftComplex>(&cbuf, width * height * sizeof(cufftComplex)))
    {
      std::cout << "[ERROR] Couldn't cudaMalloc average output" << std::endl;
      return;
    }
    if (cudaMalloc<float>(&fbuf, width * height * sizeof(float)))
    {
      cudaFree(cbuf);
      std::cout << "[ERROR] Couldn't cudaMalloc average output" << std::endl;
      return;
    }

    for (unsigned i = 0; i < nsamples; ++i)
    {
      (*average_output_)[i] = (make_average_stft_plot(cbuf, fbuf, stft_buffer, width, height, width_roi, height_roi, signal_zone, noise_zone, i, nsamples, stream));
    }

    cudaFree(cbuf);
    cudaFree(fbuf);
  }

  void ICompute::fps_count()
  {
    if (++frame_count_ >= 100)
    {
      auto time = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time - past_time_).count();
      auto manager = gui::InfoManager::get_manager();

      if (diff)
      {
        auto fps = frame_count_ * 1000 / diff;
        manager->update_info("Rendering", std::to_string(fps) + std::string(" fps"));
      }
      past_time_ = time;
      frame_count_ = 0;
    }
  }

  void ICompute::cudaMemcpyNoReturn(void* dst, const void* src, size_t size, cudaMemcpyKind kind)
  {
    ::cudaMemcpy(dst, src, size, kind);
  }

  void ICompute::autofocus_init()
  {
    // Autofocus needs to work on the same images. It will computes on copies.
    af_env_.gpu_input_size = sizeof(cufftComplex)* input_.get_pixels() * input_length_;
    cudaDestroy<cudaError_t>(&(af_env_.gpu_input_buffer_tmp));
    cudaMalloc(&af_env_.gpu_input_buffer_tmp, af_env_.gpu_input_size);

    // Wait input_length_ images in queue input_, before call make_contiguous_complex
    while (input_.get_current_elts() < input_length_)
      continue;

    // Fill gpu_input complex tmp buffer.
    make_contiguous_complex(
      input_,
      af_env_.gpu_input_buffer_tmp,
      compute_desc_.nsamples.load(),
      gpu_sqrt_vector_);

    af_env_.zone = compute_desc_.autofocus_zone;
    /* Compute square af zone. */
    const unsigned int zone_width = af_env_.zone.get_width();
    const unsigned int zone_height = af_env_.zone.get_height();

    af_env_.af_square_size = static_cast<unsigned int>(powf(2, ceilf(log2f(zone_width > zone_height ? float(zone_width) : float(zone_height)))));

    const unsigned int af_size = af_env_.af_square_size * af_env_.af_square_size;

    cudaDestroy<cudaError_t>(&(af_env_.gpu_float_buffer_af_zone));
    cudaMalloc(&af_env_.gpu_float_buffer_af_zone, af_size * sizeof(float));

    /* Initialize z_*  */
    af_env_.z_min = compute_desc_.autofocus_z_min;
    af_env_.z_max = compute_desc_.autofocus_z_max;

    const float z_div = static_cast<float>(compute_desc_.autofocus_z_div);

    af_env_.z_step = (af_env_.z_max - af_env_.z_min) / z_div;

    af_env_.af_z = 0.0f;

    af_env_.z_iter = compute_desc_.autofocus_z_iter;
    af_env_.z = af_env_.z_min;
    af_env_.focus_metric_values.clear();
  }

  void ICompute::autofocus_caller(float* input, cudaStream_t stream)
  {
    const camera::FrameDescriptor& input_fd = input_.get_frame_desc();

    frame_memcpy(input, af_env_.zone, input_fd.width, af_env_.gpu_float_buffer_af_zone, af_env_.af_square_size, stream);

    const float focus_metric_value = focus_metric(af_env_.gpu_float_buffer_af_zone, af_env_.af_square_size, stream, compute_desc_.autofocus_size);

    if (!std::isnan(focus_metric_value))
      af_env_.focus_metric_values.push_back(focus_metric_value);
    else
      af_env_.focus_metric_values.push_back(0);

    af_env_.z += af_env_.z_step;

    if (autofocus_stop_requested_ || af_env_.z > af_env_.z_max)
    {
      // Find max z
      auto biggest = std::max_element(af_env_.focus_metric_values.begin(), af_env_.focus_metric_values.end());
      const float z_div = static_cast<float>(compute_desc_.autofocus_z_div);

      /* Case the max has not been found. */
      if (biggest == af_env_.focus_metric_values.end())
        biggest = af_env_.focus_metric_values.begin();
      long long max_pos = std::distance(af_env_.focus_metric_values.begin(), biggest);

      // This is our temp max
      af_env_.af_z = af_env_.z_min + max_pos * af_env_.z_step;

      // Calculation of the new max/min, taking the old step
      af_env_.z_min = af_env_.af_z - af_env_.z_step;
      af_env_.z_max = af_env_.af_z + af_env_.z_step;

      // prepare next iter
      if (--af_env_.z_iter > 0)
      {
        af_env_.z = af_env_.z_min;
        af_env_.z_step = (af_env_.z_max - af_env_.z_min) / z_div;
        af_env_.focus_metric_values.clear();
      }
    }

    // End of the loop, free resources and notify the new z
    if (autofocus_stop_requested_ || af_env_.z_iter <= 0)
    {
      compute_desc_.zdistance = af_env_.af_z;
      compute_desc_.notify_observers();

      // if gpu_input_buffer_tmp is freed before is used by cudaMemcpyNoReturn
      cudaDestroy<cudaError_t>(&(af_env_.gpu_float_buffer_af_zone));
      cudaDestroy<cudaError_t>(&(af_env_.gpu_input_buffer_tmp));
      af_env_.focus_metric_values.clear();
      request_refresh();
    }
  }
}