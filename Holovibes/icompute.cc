#include "icompute.hh"
# include <cufft.h>

#include "fft1.cuh"
#include "fft2.cuh"
#include "stft.cuh"
#include "contrast_correction.cuh"
#include "preprocessing.cuh"
#include "autofocus.cuh"
#include "average.cuh"

namespace holovibes
{
  ICompute::ICompute(
    Queue& input,
    Queue& output,
    ComputeDescriptor& desc)
    : compute_desc_(desc)
    , input_(input)
    , output_(output)
    , gpu_stft_buffer_(nullptr)
    , gpu_stft_dup_buffer_(nullptr)
    , gpu_sqrt_vector_(nullptr)
    , gpu_lens_(nullptr)
    , plan3d_(0)
    , plan2d_(0)
    , plan1d_(0)
    , autofocus_requested_(false)
    , autocontrast_requested_(false)
    , refresh_requested_(false)
    , update_n_requested_(false)
    , stft_update_roi_requested_(false)
    , average_requested_(false)
    , average_record_requested_(false)
    , abort_construct_requested_(false)
    , termination_requested_(false)
    , average_output_(nullptr)
    , average_n_(0)
  {
    const unsigned short nsamples = desc.nsamples;

    /* if stft, we don't need to allocate more than one frame */
    if (compute_desc_.algorithm == ComputeDescriptor::STFT)
      input_length_ = 1;
    else
      input_length_ = nsamples;

    /* gpu_stft_buffer */
    cudaMalloc<cufftComplex>(&gpu_stft_buffer_,
      sizeof(cufftComplex)* compute_desc_.stft_roi_zone.load().area() * nsamples);

    /* gpu_stft_buffer */
    cudaMalloc<cufftComplex>(&gpu_stft_dup_buffer_,
      sizeof(cufftComplex)* compute_desc_.stft_roi_zone.load().area() * nsamples);

    /* Square root vector */
    cudaMalloc<float>(&gpu_sqrt_vector_, sizeof(float)* 65536);
    make_sqrt_vect(gpu_sqrt_vector_, 65535);

    /* gpu_lens */
    cudaMalloc(&gpu_lens_,
      input_.get_pixels() * sizeof(cufftComplex));

    /* CUFFT plan3d */
    if (compute_desc_.algorithm == ComputeDescriptor::FFT1
      || compute_desc_.algorithm == ComputeDescriptor::FFT2)
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
    if (compute_desc_.algorithm == ComputeDescriptor::STFT)
      cufftPlan1d(
      &plan1d_,
      nsamples,
      CUFFT_C2C,
      compute_desc_.stft_roi_zone.load().area() ? compute_desc_.stft_roi_zone.load().area() : 1
      );
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

    /* gpu_stft_dup_buffer */
    cudaFree(gpu_stft_dup_buffer_);
  }

  void ICompute::update_n_parameter(unsigned short n)
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
        << "[PREVENT_ERROR] ICompute l" << __LINE__ << " "
        << "You will reach the hard limit of cufftPlan\n"
        << compute_desc_.stft_roi_zone.load().area() * static_cast<unsigned int>(n)
        << " > "
        << 48e6
        << std::endl;
    }

    /* CUFFT plan3d realloc */
    if (plan3d_)
    {
      cufftDestroy(plan3d_) ? ++err_count : 0;
      plan3d_ = 0;
    }
    if (compute_desc_.algorithm == ComputeDescriptor::FFT1
      || compute_desc_.algorithm == ComputeDescriptor::FFT2)
      cufftPlan3d(
      &plan3d_,
      input_length_,                  // NX
      input_.get_frame_desc().width,  // NY
      input_.get_frame_desc().height, // NZ
      CUFFT_C2C) ? ++err_count : 0;

    /* CUFFT plan1d realloc */
    if (plan1d_)
    {
      cufftDestroy(plan1d_) ? ++err_count : 0;
      plan1d_ = 0;
    }
    /* gpu_stft_buffer */
    if (gpu_stft_buffer_)
      cudaFree(gpu_stft_buffer_) ? ++err_count : 0;
    gpu_stft_buffer_ = nullptr;
    /* gpu_stft_buffer */
    if (gpu_stft_dup_buffer_)
      cudaFree(gpu_stft_dup_buffer_) ? ++err_count : 0;
    gpu_stft_dup_buffer_ = nullptr;
    if (compute_desc_.algorithm == ComputeDescriptor::STFT)
    {
      cufftPlan1d(
        &plan1d_,
        n,
        CUFFT_C2C,
        compute_desc_.stft_roi_zone.load().area() ? compute_desc_.stft_roi_zone.load().area() : 1
        ) ? ++err_count : 0;

      /* gpu_stft_buffer */
      cudaMalloc<cufftComplex>(&gpu_stft_buffer_,
        sizeof(cufftComplex)* compute_desc_.stft_roi_zone.load().area() * n) ? ++err_count : 0;

      /* gpu_stft_buffer */
      cudaMalloc<cufftComplex>(&gpu_stft_dup_buffer_,
        sizeof(cufftComplex)* compute_desc_.stft_roi_zone.load().area() * n) ? ++err_count : 0;
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

  void ICompute::request_refresh()
  {
    refresh_requested_ = true;
  }

  void ICompute::request_float_output(std::string& file_src, const unsigned int nb_frame)
  {
    try
    {
      float_output_file_.open(file_src, std::ofstream::trunc | std::ofstream::binary);
      float_output_nb_frame_ = nb_frame;
      float_output_requested_ = true;
      request_refresh();
      std::cout << "[ICompute]: float record start." << std::endl;
    }
    catch (std::exception& e)
    {
      std::cout << "[ICompute]: float record: " << e.what() << std::endl;
      request_float_output_stop();
    }
  }

  void ICompute::request_float_output_stop()
  {
    if (float_output_file_.is_open())
      float_output_file_.close();
    float_output_requested_ = false;
    request_refresh();
    std::cout << "[ICompute]: float record done." << std::endl;
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

  void ICompute::request_average(
    ConcurrentDeque<std::tuple<float, float, float>>* output)
  {
    assert(output != nullptr);

    if (compute_desc_.algorithm == ComputeDescriptor::STFT)
      output->resize(compute_desc_.nsamples.load());
    average_output_ = output;

    average_requested_ = true;
    request_refresh();
  }

  void ICompute::request_average_record(
    ConcurrentDeque<std::tuple<float, float, float>>* output,
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

  /* Looks like the ICompute, but it searches for the right z value.
  The method choosen, iterates on the numbers of points given by the user
  between min and max, take the max and increase the precision to focus
  around the max more and more according to the number of iterations
  choosen.
  If the users chooses 2 iterations, a max will be choosen out of 10 images
  done between min and max. This max will be added and substracted a value
  in order to have a new (more accurate) zmin and zmax. And 10 more images
  will be produced, giving a better zmax
  */

  void ICompute::autofocus_caller()
  {
    // TODO: AHAHAHAHAHAHAHHH
  }
}