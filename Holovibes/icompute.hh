/*! \file 
 * 
 * Stores functions helping the editing of the images. */
#pragma once

# include <fstream>
# include <cufft.h>
# include <chrono>
# include <memory>

# include "config.hh"
# include "pipeline_utils.hh"
# include "geometry.hh"

/* Forward declarations. */
namespace holovibes
{
  struct UnwrappingResources;
  class Queue;
  template <class T> class ConcurrentDeque;
  class ComputeDescriptor;
}

/* Forward declarations. */
namespace holovibes
{
  struct UnwrappingResources;
}

namespace holovibes
{
/* \brief Stores functions helping the editing of the images.
 *
 * Stores all the functions that will be used before doing
 * any sort of editing to the image (i.e. refresh functions
 * or caller).
 */
  class ICompute
  {
    friend class ThreadCompute;
  public:

    /*! \brief Contains all the information related to the autofocus. */
    struct af_env
    {
      float           z;
      float           z_min;
      float           z_max;
      float           z_step;
      unsigned int    z_iter;
      float           af_z;
      std::vector<float> focus_metric_values;
      Rectangle       zone;
      float*          gpu_float_buffer_af_zone;
      cufftComplex*   gpu_input_buffer_tmp;
      size_t          gpu_input_size;
      unsigned int    af_square_size;
    };

    /* \brief Pre-allocation needed ressources for the autofocus to work. */
    void autofocus_init();

    /* \brief Simple wrapper around cudaMemcpy. */
    void cudaMemcpyNoReturn(void* dst, const void* src, size_t size, cudaMemcpyKind kind);

    /*! \brief Construct the ICompute object with 2 queues and 1 compute desc. */
    ICompute(
      Queue& input,
      Queue& output,
      ComputeDescriptor& desc);

    /*! \brief Destroy the ICompute object. */
    virtual ~ICompute();

    /*! \{ \name ICompute request methods */
    /*! \brief Request the ICompute to refresh. */
    void request_refresh();

    /*! \brief Request the ICompute to apply the autofocus algorithm. */
    void request_autofocus();

    /*! \brief Request the ICompute to stop the occuring autofocus. */
    void request_autofocus_stop();

    /*! \brief Request the ICompute to apply the autocontrast algorithm. */
    void request_autocontrast();

    /*! \brief Request the ICompute to apply the stft algorithm in the border. And call request_update_n */
    void request_stft_roi_update();

    /*! \brief Request the ICompute to apply the stft algorithm in full window. And call request_update_n */
    void request_stft_roi_end();

    /*! \brief Request the ICompute to update the nsamples parameter.
    *
    * Use this method when the user has requested the nsamples parameter to be
    * updated. The ICompute will automatically resize FFT buffers to contains
    * nsamples frames. */
    void request_update_n(const unsigned short n);

    /*! Set the size of the unwrapping history window. */
    void request_update_unwrap_size(const unsigned size);

    /*! The boolean will determine activation/deactivation of the unwrapping phase
     * in the unwrap() function. */
    void request_unwrapping(const bool value);

    /*! \brief Request the ICompute to fill the output vector.
    *
    * \param output std::vector to fill with (average_signal, average_noise,
    * average_ratio).
    * \note This method is only used by the GUI to draw the average graph. */
    void request_average(
      ConcurrentDeque<std::tuple<float, float, float, float>>* output);
    /*! \brief Request the ICompute to stop the average compute. */
    void request_average_stop();

    /*! \brief Request the ICompute to fill the output vector with n samples.
    *
    * \param output std::vector to fill with (average_signal, average_noise,
    * average_ratio).
    * \param n number of samples to record.
    * \note This method is used to record n samples and then automatically
    * refresh the ICompute. */
    void request_average_record(
      ConcurrentDeque<std::tuple<float, float, float, float>>* output,
      const unsigned int n);

    /*! \brief Request the ICompute to start record gpu_float_buf_ (Stop output). */
    void request_float_output(Queue* fqueue);

    /*! \brief Request the ICompute to stop the record gpu_float_buf_ (Relaunch output). */
    void request_float_output_stop();

    /*! \brief Ask for the end of the execution loop. */
    void request_termination();
    /*! \} */ // End of requests group.

    /*! \brief Return true while ICompute is recording float. */
    bool is_requested_float_output() const
    {
      return (float_output_requested_);
    }

    /*! \brief Execute one iteration of the ICompute.
    *
    * * Checks the number of frames in input queue that must at least
    * nsamples*.
    * * Call each function of the ICompute.
    * * Enqueue the output frame contained in gpu_output_buffer.
    * * Dequeue one frame of the input queue.
    * * Check if a ICompute refresh has been requested.
    *
    * The ICompute can not be interrupted for parameters changes until the
    * refresh method is called. */
    virtual void exec() = 0;

  protected:
    /*! \brief Generate the ICompute vector. */
    virtual void refresh();

    /*! \brief Realloc all buffer with the new nsamples and update ICompute */
    virtual void update_n_parameter(unsigned short n);

    /*! \{ \name caller methods (helpers)
    *
    * For some features, it might be necessary to do special treatment. For
    * example, store a returned value in a std::vector. */

    /*! \brief Call autocontrast algorithm and then update the compute
    * descriptor. */
    static void autocontrast_caller(
      float* input,
      const unsigned int size,
      ComputeDescriptor& compute_desc,
      cudaStream_t stream);

    /*! \see request_average
    * \brief Call the average algorithm and store the result in the vector.
    * \param input Input float frame pointer
    * \param width Width of the input frame
    * \param height Height of the input frame
    * \param signal Signal zone
    * \param noise Noise zone */
    void average_caller(
      float* input,
      const unsigned int width,
      const unsigned int height,
      const Rectangle& signal,
      const Rectangle& noise,
      cudaStream_t stream);

    /*! \see request_average_record
    * \brief Call the average algorithm, store the result and count n
    * iterations. Request the ICompute to refresh when record is over.
    * \param input Input float frame pointer
    * \param width Width of the input frame
    * \param height Height of the input frame
    * \param signal Signal zone
    * \param noise Noise zone */
    void average_record_caller(
      float* input,
      const unsigned int width,
      const unsigned int height,
      const Rectangle& signal,
      const Rectangle& noise,
      cudaStream_t stream);

    /*! \see request_average
    * \brief For nsamples in input, reconstruct image,
    * clear previous result, call the average algorithm and store each result
    * \param input Input buf, contain nsamples bursting frame
    * \param width Width of one frame
    * \param height Height of one frame
    * \param signal Signal zone
    * \param noise Noise zone */
    void average_stft_caller(
      cufftComplex* input,
      const unsigned int width,
      const unsigned int height,
      const unsigned int width_roi,
      const unsigned int height_roi,
      Rectangle& signal_zone,
      Rectangle& noise_zone,
      const unsigned int nsamples,
      cudaStream_t stream);

    /*! \see request_autofocus
    * \brief Autofocus caller looks like the ICompute refresh method.
    *
    * The autofocus caller generates multiple holograms (with variable z) on the
    * same image set. Computes the focus_metric on each hologram and sets the
    * proper value of z in ComputeDescriptor. */
    virtual void autofocus_caller(float* input, cudaStream_t stream);
    /*! \} */ // End of callers group

    /*! \brief Add frame in fqueue_. */
    void record_float(float* float_output, cudaStream_t stream);

    /*! \brief Print fps each 100 frames
    **
    ** Use InfoManager */
    void fps_count();

    /*! \{ \name Disable copy/assignments. */
    ICompute& operator=(const ICompute&) = delete;
    ICompute(const ICompute&) = delete;
    /*! \} */

  protected:
    /*! \brief Shared (GUI/CLI) ComputeDescriptor */
    ComputeDescriptor& compute_desc_;
    /*! \brief Input frame queue : 16-bit frames. */
    Queue& input_;
    /*! \brief Output frame queue : 16-bit frames. */
    Queue& output_;

    /*! Vector filled with sqrtf values. */
    float* gpu_sqrt_vector_;

    /*! All buffers needed for phase unwrapping are here. */
    std::shared_ptr<UnwrappingResources> unwrap_res_;

    /*! cufftComplex array containing n contiguous ROI of frames. */
    cufftComplex* gpu_stft_buffer_;
    /*! cufftComplex array containing save of n contiguous ROI of frames. */
	cufftComplex* gpu_stft_dup_buffer_;
	/*! cufftComplex array containing lens. */
	cufftComplex* gpu_lens_;
	/*! cufftComplex array containing kernel. */
	float* gpu_kernel_buffer_;
	/*! cufftComplex array containing tmp input. */
	cufftComplex* gpu_tmp_input_;
	/*! cufftComplex queue */
	cufftComplex* gpu_special_queue_;
	unsigned int  gpu_special_queue_start_index;
	unsigned int  gpu_special_queue_max_index;
    /*! CUDA FFT Plan 3D. Set to a specific CUDA stream in Pipe and Pipeline. */
    cufftHandle plan3d_;
    /*! CUDA FFT Plan 2D. Set to a specific CUDA stream in Pipe and Pipeline. */
    cufftHandle plan2d_;
    /*! CUDA FFT Plan 1D. Set to a specific CUDA stream in Pipe and Pipeline. */
    cufftHandle plan1d_;
    /*! \} */
    /*! \{ \name request flags */
    bool unwrap_requested_;
    bool autofocus_requested_;
    bool autofocus_stop_requested_;
    bool autocontrast_requested_;
    bool refresh_requested_;
    bool update_n_requested_;
    bool stft_update_roi_requested_;
    bool average_requested_;
    bool average_record_requested_;
    bool float_output_requested_;
    bool abort_construct_requested_;
    bool termination_requested_;
    /*! \} */

    /*! \brief Number of frame in input. */
    unsigned int input_length_;
    /*! \brief Float queue for float record */
    Queue*       fqueue_;
    /*! \brief index of current element trait in stft */
    unsigned int curr_elt_stft_;
    /*! \brief Containt q image for vibrometry in stft mode
    *
    * Is only allocated if stft mode and vibrometry is enable */
    cufftComplex* q_gpu_stft_buffer_;

    /*! \{ \name average plot */
    ConcurrentDeque<std::tuple<float, float, float, float>>* average_output_;
    unsigned int average_n_;
    /*! \} */
    /*! \{ \name fps_count */
    std::chrono::time_point<std::chrono::system_clock, std::chrono::system_clock::duration> past_time_;
    unsigned int frame_count_;
    /*! \} */
    /*! \brief containt all var needed by auto_focus */
    af_env        af_env_;
    /*! \brief Buffer use to read gpu buffer and write it to float_output_file_*/
    float*        cpu_float_buffer_;
    /*! \brief Ofstream use by float_output_recorder. */
    std::ofstream float_output_file_;
  };
}