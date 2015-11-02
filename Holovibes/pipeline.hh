#ifndef PIPELINE_HH
# define PIPELINE_HH

# include <vector>
# include <tuple>
# include <functional>
# include <cufft.h>

# include "queue.hh"
# include "concurrent_deque.hh"
# include "compute_descriptor.hh"

# include <fstream>

namespace holovibes
{
  /*! \brief Pipeline is a class that applies treatments on input frames.
   *
   * # Why doing this way ?
   *
   * The goal of the pipeline is to build a vector filled with functions to
   * apply on frames. This way it avoids to have a monolithic method plenty of
   * if/else following what the user wants to do. In most cases, the treatment
   * remains the same at runtime, most jump conditions will always be the same.
   *
   * When the pipeline is refreshing, the vector is updated with last user
   * parameters. Keep in mind that the software is incredibly faster than user
   * inputs in GUI, so treatments are always applied with the same parameters.
   *
   * ## RAII
   *
   * The pipeline manages almost every CPU/GPU memory ressources. Once again,
   * most of frames buffer will always keep the same size, so it is not
   * necessary to allocate memory with malloc/cudaMalloc in each treatment
   * functions. Keep in mind, malloc is costly !
   *
   * ## Request system
   *
   * In order to avoid strange concurrent behaviours, the pipeline is used with
   * a request system. When the compute descriptor is modified the GUI will
   * request the pipeline to refresh with updated parameters.
   *
   * Also, some events such as autofocus or autoconstrast will be executed only
   * for one iteration. For example, request_autocontrast will add the autocontrast
   * algorithm in the pipeline and will automatically set a pipeline refresh so
   * that the autocontrast algorithm will be done only once.
   */
  class Pipeline
  {
    /*! \brief Vector of procedures type */
    using FnVector = std::vector < std::function<void()> >;
    friend class ThreadCompute;
  public:
    /*! \brief Allocate CPU/GPU ressources for computation.
     * \param input Input queue containing acquired frames.
     * \param output Output queue where computed frames will be stored.
     * \param desc ComputeDescriptor that contains computation parameters. */
    Pipeline(
      Queue& input,
      Queue& output,
      ComputeDescriptor& desc);
    virtual ~Pipeline();

    /*! \{ \name Pipeline request methods */
    /*! \brief Request the pipeline to refresh. */
    void request_refresh();
    /*! \brief Request the pipeline to apply the autofocus algorithm. */
    void request_autofocus();
    /*! \brief Request the pipeline to stop the occuring autofocus. */
    void request_autofocus_stop();
    /*! \brief Request the pipeline to apply the autocontrast algorithm. */
    void request_autocontrast();
    /*! \brief Request the pipeline to apply the stft algorithm in the border. And call request_update_n */
    void request_stft_roi_update();
    /*! \brief Request the pipeline to apply the stft algorithm in full window. And call request_update_n */
    void request_stft_roi_end();
    /*! \brief Request the pipeline to update the nsamples parameter.
     *
     * Use this method when the user has requested the nsamples parameter to be
     * updated. The pipeline will automatically resize FFT buffers to contains
     * nsamples frames. */
    void request_update_n(unsigned short n);
    /*! \brief Request the pipeline to fill the output vector.
     *
     * \param output std::vector to fill with (average_signal, average_noise,
     * average_ratio).
     * \note This method is only used by the GUI to draw the average graph. */
    void request_average(
      ConcurrentDeque<std::tuple<float, float, float>>* output);

    /*! \brief Request the pipeline to fill the output vector with n samples.
     *
     * \param output std::vector to fill with (average_signal, average_noise,
     * average_ratio).
     * \param n number of samples to record.
     * \note This method is used to record n samples and then automatically
     * refresh the pipeline. */
    void request_average_record(
      ConcurrentDeque<std::tuple<float, float, float>>* output,
      unsigned int n);
    /*! \} */

    /*! \brief Request the pipeline to start record gpu_float_buf_ (Stop output). */
    void request_float_output(std::string& file_src, unsigned int nb_frame);
    /*! \brief Request the pipeline to stop the record gpu_float_buf_ (Relaunch output). */
    void request_float_output_stop();

    /*! \brief Return true while pipeline is recording float. */
    bool is_requested_float_output() const
    {
      return (float_output_requested_);
    }

    /*! \brief Execute one iteration of the pipeline.
     *
     * * Checks the number of frames in input queue that must at least
     * nsamples*.
     * * Call each function of the pipeline.
     * * Enqueue the output frame contained in gpu_output_buffer.
     * * Dequeue one frame of the input queue.
     * * Check if a pipeline refresh has been requested.
     *
     * The pipeline can not be interrupted for parameters changes until the
     * refresh method is called. */
    void exec();
  private:
    void update_n_parameter(unsigned short n);

    /*! \{ \name caller methods (helpers)
     *
     * For some features, it might be necessary to do special treatment. For
     * example, store a returned value in a std::vector. */

    /*! \brief Call autocontrast algorithm and then update the compute
     * descriptor. */
    static void autocontrast_caller(
      float* input,
      unsigned int size,
      ComputeDescriptor& compute_desc);

    /*! \see request_average
     * \brief Call the average algorithm and store the result in the vector.
     * \param input Input float frame pointer
     * \param width Width of the input frame
     * \param height Height of the input frame
     * \param signal Signal zone
     * \param noise Noise zone */
    void average_caller(
      float* input,
      unsigned int width,
      unsigned int height,
      Rectangle& signal,
      Rectangle& noise);
    /*! \see request_average_record
     * \brief Call the average algorithm, store the result and count n
     * iterations. Request the pipeline to refresh when record is over.
     * \param input Input float frame pointer
     * \param width Width of the input frame
     * \param height Height of the input frame
     * \param signal Signal zone
     * \param noise Noise zone */
    void average_record_caller(
      float* input,
      unsigned int width,
      unsigned int height,
      Rectangle& signal,
      Rectangle& noise);
    /*! \see request_average
     * \brief For nsamples in input, reconstruct image,
     * clear previous result, call the average algorithm and store each result
     * \param input Input buf, contain nsamples bursting frame
     * \param width Width of one frame
     * \param height Height of one frame
     * \param signal Signal zone
     * \param noise Noise zone */
    void average_stft_caller(
      cufftComplex*    input,
      unsigned int     width,
      unsigned int     height,
      unsigned int     width_roi,
      unsigned int     height_roi,
      Rectangle&       signal_zone,
      Rectangle&       noise_zone,
      unsigned int     nsamples
      );
    /*! \see request_autofocus
     * \brief Autofocus caller looks like the pipeline refresh method.
     *
     * The autofocus caller generates multiple holograms (with variable z) on the
     * same image set. Computes the focus_metric on each hologram and sets the
     * proper value of z in ComputeDescriptor. */
    void autofocus_caller();
    /*! \} */

    /*! \brief Generate the pipeline vector. */
    void refresh();

    /*! \brief Record one frame in gpu_float_buf_ to file_. */
    void record_float();

    /*! \{ \name Disable copy/assignments. */
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(const Pipeline&) = delete;
    /*! \} */
  private:
    /*! \brief Core of the pipeline */
    FnVector fn_vect_;
    /*! \brief Shared (GUI/CLI) ComputeDescriptor */
    ComputeDescriptor& compute_desc_;
    /*! \brief Input frame queue : 16-bit frames. */
    Queue& input_;
    /*! \brief Output frame queue : 16-bit frames. */
    Queue& output_;

    /*! \{ \name Memory buffers
     * \brief Memory buffers pointers
     *
     * * fields with gpu prefix are allocated in GPU memory
     * * fields with cpu prefix are allocated in CPU memory */
    /*! cufftComplex array containing n contiguous frames. */
    cufftComplex* gpu_input_buffer_;
    /*! Intermediate buffer containing nsamples frames converted to complex;
    **  shifted by one frame at each iteration, keeping nsamples - 1 already
    **  converted frames. */
    cufftComplex* gpu_complex_buffer_;
    /*! cufftComplex array containing n contiguous ROI of frames. */
    cufftComplex* gpu_stft_buffer_;
    /*! cufftComplex array containing save of n contiguous ROI of frames. */
    cufftComplex* gpu_stft_dup_buffer_;
    /*! Output frame containing n frames ordered in frequency. */
    unsigned short* gpu_output_buffer_;
    /*! GPU float frame */
    float* gpu_float_buffer_;
    /*! Vector filled with sqrtf values. */
    float* gpu_sqrt_vector_;
    /*! cufftComplex array containing lens. */
    cufftComplex* gpu_lens_;
    /*! CUDA FFT Plan 3D. */
    cufftHandle plan3d_;
    /*! CUDA FFT Plan 2D. */
    cufftHandle plan2d_;
    /*! CUDA FFT Plan 1D. */
    cufftHandle plan1d_;
    /*! \} */

    /*! Input frame pointer. */
    cufftComplex* gpu_input_frame_ptr_;

    /*! \{ \name request flags */
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
    /*! \} */

    /*! \brief Number of frame in input. */
    unsigned int input_length_;
    /*! \brief Number of frame to record before request_float_output_stop. */
    unsigned int float_output_nb_frame_;
    /*! \brief index of current element trait in stft */
    unsigned int curr_elt_stft_;
    /*! \brief Ofstream use by float_output_recorder. */
    std::ofstream float_output_file_;

    /*! \{ \name average plot */
    ConcurrentDeque<std::tuple<float, float, float>>* average_output_;
    unsigned int average_n_;
    /*! \} */
  };
}

#endif /* !PIPELINE_HH */