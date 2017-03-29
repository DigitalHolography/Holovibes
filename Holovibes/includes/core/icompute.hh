/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
 * 
 * Stores functions helping the editing of the images. */
#pragma once

# include <fstream>
# include <cufft.h>
# include <chrono>
# include <mutex>
# include <memory>
# include <atomic>

# include "config.hh"
# include "pipeline_utils.hh"
# include "Rectangle.hh"
# include "observable.hh"
//# include "sMainWindow.hh"

/* Forward declarations. */
namespace holovibes
{
  struct UnwrappingResources;
  struct UnwrappingResources_2d;
  class Queue;
  template <class T> class ConcurrentDeque;
  class ComputeDescriptor;
}

namespace holovibes
{
/* \brief Stores functions helping the editing of the images.
 *
 * Stores all the functions that will be used before doing
 * any sort of editing to the image (i.e. refresh functions
 * or caller).
 */
  class ICompute : public Observable
  {
    friend class ThreadCompute;
  public:

    /*! \brief Contains all the information related to the autofocus. */
    struct af_env
    {
      float					z;
      float					z_min;
      float					z_max;
      float					z_step;
      unsigned int			z_iter;
      float					af_z;
      std::vector<float>	focus_metric_values;
		gui::Rectangle		zone;
		float				*gpu_float_buffer_af_zone;
		cufftComplex		*gpu_input_buffer_tmp;
		size_t				gpu_input_size;
		unsigned int		af_square_size;
    };

	/* these states are used for take ref when we need to do action punctually in the pipe*/
	enum state
	{
		ENQUEUE,
		COMPUTE
	};

    /* \brief Pre-allocation needed ressources for the autofocus to work. */
    void autofocus_init();

    /*! \brief Construct the ICompute object with 2 queues and 1 compute desc. */
    ICompute(
      Queue& input,
      Queue& output,
      ComputeDescriptor& desc);

    /*! \brief Destroy the ICompute object. */
    virtual ~ICompute();

	/*! \brief Realloc the image accumulation buffer */
	void update_acc_parameter();

	/*! \brief Realloc the images_reference_buffer */
	void update_ref_diff_parameter();

    /*! \{ \name ICompute request methods */
    /*! \brief Request the ICompute to refresh. */
    void request_refresh();

	/*! \brief Request to refresh the accumulation queue. */
	void request_acc_refresh();

	/*! \brief Request to refresh the reference queue. */
	void request_ref_diff_refresh();

    /*! \brief Request the ICompute to apply the autofocus algorithm. */
    void request_autofocus();

    /*! \brief Request the ICompute to stop the occuring autofocus. */
    void request_autofocus_stop();

    /*! \brief Request the ICompute to apply the autocontrast algorithm. */
    void request_autocontrast();

    /*! \brief Request the ICompute to apply the stft algorithm in the border. And call request_update_n */
    void request_filter2D_roi_update();

    /*! \brief Request the ICompute to apply the stft algorithm in full window. And call request_update_n */
    void request_filter2D_roi_end();

    /*! \brief Request the ICompute to update the nsamples parameter.
    *
    * Use this method when the user has requested the nsamples parameter to be
    * updated. The ICompute will automatically resize FFT buffers to contains
    * nsamples frames. */
    void request_update_n(const unsigned short n);

    /*! Set the size of the unwrapping history window. */
    void request_update_unwrap_size(const unsigned size);

    /*! The boolean will determine activation/deactivation of the unwrapping 1d */
    void request_unwrapping_1d(const bool value);

	/*! The boolean will determine activation/deactivation of the unwrapping 2d */
	void request_unwrapping_2d(const bool value);

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

	/*! \brief Request the ICompute to start record gpu_float_buf_ (Stop output). */
	void request_complex_output(Queue* fqueue);

	/*! \brief Request the ICompute to stop the record gpu_float_buf_ (Relaunch output). */
	void request_complex_output_stop();

	/*! \brief Add current img to img_phase queue*/
	void queue_enqueue(void* input, Queue* queue);
	
	/* \brief handle all the stft workaround. Allow us to use a counter to compute STFT asynchronously */
	void stft_handler(cufftComplex* input, cufftComplex* output);

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

	void	create_stft_slice_queue();
	void	delete_stft_slice_queue();
	void	update_stft_slice_queue();
	Queue&	get_stft_slice_queue(int i);
	bool	get_cuts_request();
	bool	get_cuts_delete_request();
	bool	get_request_refresh();

  protected:
    /*! \brief Generate the ICompute vector. */
    virtual void refresh();

	/*! \brief In case an allocation error occured */
	virtual void allocation_failed(const int& err_count, std::exception& e);

    /*! \brief Realloc all buffer with the new nsamples and update ICompute */
    virtual bool update_n_parameter(unsigned short n);

    /*! \{ \name caller methods (helpers)
    *
    * For some features, it might be necessary to do special treatment. For
    * example, store a returned value in a std::vector. */

    /*! \brief Call autocontrast algorithm and then update the compute
    * descriptor. */
    static void autocontrast_caller(float				*input,
									const uint			size,
									ComputeDescriptor&	compute_desc,
									std::atomic<float>&	min,
									std::atomic<float>&	max,
									cudaStream_t		stream);

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
      const gui::Rectangle& signal,
      const gui::Rectangle& noise,
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
      const gui::Rectangle& signal,
      const gui::Rectangle& noise,
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
      gui::Rectangle& signal_zone,
      gui::Rectangle& noise_zone,
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

	/*! \brief Add frame in fqueue_. */
	void record_complex(cufftComplex* complex_output, cudaStream_t stream);

	/* \brief handle all the reference workaround when take_ref button is pushed. */
	void handle_reference(cufftComplex* input, const unsigned int nframes);

	/* /* \brief handle all the reference workaround when slinding button is pushed. */ 
	void handle_sliding_reference(cufftComplex* input, const unsigned int nframes);

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

    /*! All buffers needed for phase unwrapping are here. */
    std::shared_ptr<UnwrappingResources> unwrap_res_;

	/*! All buffers needed for phase unwrapping 2d are here. */
	std::shared_ptr<UnwrappingResources_2d> unwrap_res_2d_;

    /*! cufftComplex array containing n contiguous ROI of frames. */
    cufftComplex *gpu_stft_buffer_;
	std::mutex	stftGuard;

	/*! cufftComplex array containing lens. */
	cufftComplex *gpu_lens_;
	/*! cufftComplex array containing kernel. */
	float *gpu_kernel_buffer_;
	/*! cufftComplex array containing tmp input. */
	cufftComplex *gpu_tmp_input_;
	/*! cufftComplex queue */
	cufftComplex *gpu_special_queue_;
	unsigned int  gpu_special_queue_start_index;
	unsigned int  gpu_special_queue_max_index;
    /*! CUDA FFT Plan 3D. Set to a specific CUDA stream in Pipe and Pipeline. */
    cufftHandle plan3d_;
    /*! CUDA FFT Plan 2D. Set to a specific CUDA stream in Pipe and Pipeline. */
    cufftHandle plan2d_;
    /*! CUDA FFT Plan 1D. Set to a specific CUDA stream in Pipe and Pipeline. */
    cufftHandle plan1d_;

	/*! CUDA FFT Plan 1D. Set to a specific CUDA stream in Pipe and Pipeline. */
	cufftHandle plan1d_stft_;

    /*! \} */

    /*! \brief Number of frame in input. */
    unsigned int input_length_;
    /*! \brief Float queue for float record */
    Queue *fqueue_;
    /*! \brief index of current element trait in stft */
    unsigned int curr_elt_stft_;

    ConcurrentDeque<std::tuple<float, float, float, float>>* average_output_;
    unsigned int average_n_;
    /*! \} */
    /*! \{ \name fps_count */
    std::chrono::time_point<std::chrono::steady_clock> past_time_;
  
	unsigned int frame_count_;
    /*! \} */
    /*! \brief containt all var needed by auto_focus */
    af_env        af_env_;

	/*! \brief Queue for phase accumulation*/
	Queue *gpu_img_acc_;

	/*! \brief Queue for stft */
	Queue *gpu_stft_queue_;
	Queue *gpu_stft_slice_queue_xz;
	Queue *gpu_stft_slice_queue_yz;
	
	/* \brief Queue for the reference diff */
	Queue *gpu_ref_diff_queue_;

	/* these states are used for take ref when we need to do action punctually in the pipe*/
	enum state ref_diff_state_;

	cufftComplex *gpu_filter2d_buffer;

	unsigned int ref_diff_counter;

	unsigned int stft_frame_counter;

	int		slice_width_;
	int		slice_height_;
	int		slice_depth_;
	int		frame_res_xz_;
	int		frame_res_yz_;

	/*! \{ \name request flags */
	bool unwrap_1d_requested_;
	bool unwrap_2d_requested_;
	bool autofocus_requested_;
	bool autofocus_stop_requested_;
	bool autocontrast_requested_;
	bool refresh_requested_;
	bool update_n_requested_;
	bool stft_update_roi_requested_;
	bool average_requested_;
	bool average_record_requested_;
	bool float_output_requested_;
	bool complex_output_requested_;
	bool abort_construct_requested_;
	bool termination_requested_;
	bool update_acc_requested_;
	bool update_ref_diff_requested_;
	bool request_stft_cuts_;
	bool request_delete_stft_cuts_;
	/*! \} */
  };
}