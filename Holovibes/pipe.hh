#pragma once

# include <tuple>

# include "icompute.hh"

namespace holovibes
{
  /*! \brief Pipe is a class that applies treatments on input frames.
   *
   * # Why doing this way ?
   *
   * The goal of the pipe is to build a vector filled with functions to
   * apply on frames. This way it avoids to have a monolithic method plenty of
   * if/else following what the user wants to do. In most cases, the treatment
   * remains the same at runtime, most jump conditions will always be the same.
   *
   * When the pipe is refreshing, the vector is updated with last user
   * parameters. Keep in mind that the software is incredibly faster than user
   * inputs in GUI, so treatments are always applied with the same parameters.
   *
   * ## RAII
   *
   * The pipe manages almost every CPU/GPU memory ressources. Once again,
   * most of frames buffer will always keep the same size, so it is not
   * necessary to allocate memory with malloc/cudaMalloc in each treatment
   * functions. Keep in mind, malloc is costly !
   *
   * ## Request system
   *
   * In order to avoid strange concurrent behaviours, the pipe is used with
   * a request system. When the compute descriptor is modified the GUI will
   * request the pipe to refresh with updated parameters.
   *
   * Also, some events such as autofocus or autoconstrast will be executed only
   * for one iteration. For example, request_autocontrast will add the autocontrast
   * algorithm in the pipe and will automatically set a pipe refresh so
   * that the autocontrast algorithm will be done only once.
   */
  class Pipe : public ICompute
  {
  public:
    /*! \brief Allocate CPU/GPU ressources for computation.
     * \param input Input queue containing acquired frames.
     * \param output Output queue where computed frames will be stored.
     * \param desc ComputeDescriptor that contains computation parameters. */
    Pipe(
      Queue& input,
      Queue& output,
      ComputeDescriptor& desc);

    virtual ~Pipe();

  protected:
    /*! \brief Generate the ICompute vector. */
    virtual void refresh();

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
    virtual void exec();

    /*! \brief Realloc all buffer with the new nsamples and update ICompute */
    virtual void update_n_parameter(unsigned short n);

    virtual void autofocus_caller();
  private:
    /*! \brief Core of the pipe */
    FnVector fn_vect_;

    /*! \{ \name Memory buffers
     * \brief Memory buffers pointers
     *
     * * fields with gpu prefix are allocated in GPU memory
     * * fields with cpu prefix are allocated in CPU memory
     * * fields cufftHandle are allocated in GPU memory */
    /*! cufftComplex array containing n contiguous frames. */
    cufftComplex* gpu_input_buffer_;
    /*! Output frame containing n frames ordered in frequency. */
    unsigned short* gpu_output_buffer_;
    /*! GPU float frame */
    float* gpu_float_buffer_;

    /*! Input frame pointer. */
    cufftComplex* gpu_input_frame_ptr_;
  };
}