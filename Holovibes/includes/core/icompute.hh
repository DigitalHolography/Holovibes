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

# include "config.hh"
# include "pipeline_utils.hh"
# include "rect.hh"
# include "observable.hh"
# include "gpib_controller.hh"
# include "frame_desc.hh"
# include "unique_ptr.hh"
# include "cufft_handle.hh"

namespace holovibes
{
	using	Tuple4f = std::tuple<float, float, float, float>;
	class Queue;
	template <class T> class ConcurrentDeque;
	class ComputeDescriptor;
	enum WindowKind;
}

namespace holovibes
{
	/*! \brief Struct containing main buffers used by the pipe. */
	struct CoreBuffers
	{
		/** Input buffer. Contains only one frame. We fill it with the input frame, then with the correct computed p frame. */
		cuda_tools::UniquePtr<cufftComplex>		gpu_input_buffer_ = nullptr;

		/** Float buffer. Contains only one frame. We fill it with the correct computed p frame converted to float. */
		cuda_tools::UniquePtr<float>			gpu_float_buffer_ = nullptr;
		/** Size in components (size in byte / sizeof(float)) of the gpu_float_buffer_.
		 Could be removed by changing gpu_float_buffer_ type to cuda_tools::Array. */
		uint									gpu_float_buffer_size_ = 0;
		/** Float XZ buffer. Contains only one frame. We fill it with the correct computed p XZ frame.
		 It is of void type because it is also used for complex slices.
		 Could be better to have an other buffer used only for complex slices. */
		cuda_tools::UniquePtr<void>				gpu_float_cut_xz_ = nullptr;
		/** Float YZ buffer. Contains only one frame. We fill it with the correct computed p YZ frame. */
		cuda_tools::UniquePtr<void>				gpu_float_cut_yz_ = nullptr;

		/** Unsigned Short output buffer. Contains only one frame, inserted after all postprocessing on float_buffer */
		cuda_tools::UniquePtr<unsigned short>	gpu_output_buffer_ = nullptr;
		/** Unsigned Short XZ output buffer. Contains only one frame, inserted after all postprocessing on float_buffer_cut_xz */
		cuda_tools::UniquePtr<unsigned short>	gpu_ushort_cut_xz_ = nullptr;
		/** Unsigned Short YZ output buffer. Contains only one frame, inserted after all postprocessing on float_buffer_cut_yz */
		cuda_tools::UniquePtr<unsigned short>	gpu_ushort_cut_yz_ = nullptr;

		/***/
		cuda_tools::UniquePtr<void>				gpu_complex_buffer_xy_ = nullptr;

		/**contain only one frame used only for convolution*/
		cuda_tools::UniquePtr<float>			gpu_convolution_buffer_ = nullptr;
	};

	/*! \brief Struct containing variables related to STFT shared by multiple features of the pipe. */
	struct Stft_env
	{
		/** Mutex used for every temporal fft computation (fft or plan computation).
		 TODO: Check if it is really usefull (if there is really more than one thread using these variables). */
		std::mutex							stftGuard_;

		/** STFT Queue. Constains nSize frames. It accumulates input frames after spatial fft,
		 in order to apply STFT only when the frame counter is equal to STFT steps. */
		std::unique_ptr<Queue>				gpu_stft_queue_ = nullptr;
		/** STFT buffer. Contains nSize frames. Contains the result of the STFT done on the STFT queue. */
		cuda_tools::UniquePtr<cufftComplex>	gpu_stft_buffer_ = nullptr;
		/** STFT XZ Queue. Contains the ouput of the STFT on slice XZ. Enqueued with gpu_float_buffer or gpu_ushort_buffer. */
		std::unique_ptr<Queue>				gpu_stft_slice_queue_xz = nullptr;
		/** STFT YZ Queue. Contains the ouput of the STFT on slice YZ. Enqueued with gpu_float_buffer or gpu_ushort_buffer. */
		std::unique_ptr<Queue>				gpu_stft_slice_queue_yz = nullptr;
		/** Plan 1D used for the STFT. */
		cuda_tools::CufftHandle				plan1d_stft_;

		/** Boolean set if the STFT has been performed during this pipe iteration. Used to not re-compute post-processing between STFT Steps. */
		bool								stft_handle_ = false;
		/** Frame Counter. Counter before the next STFT perform. */
		uint								stft_frame_counter_ = 1;
	};

	/** \brief Structure containing variables related to the average computation and recording. */
	struct Average_env
	{
		ConcurrentDeque<Tuple4f>* average_output_ = nullptr;
		uint	average_n_ = 0;
	};

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

		ICompute(
			Queue& input,
			Queue& output,
			ComputeDescriptor& desc);
		virtual ~ICompute();

		void request_refresh();
		void request_acc_refresh();
		void request_ref_diff_refresh();
		void request_autofocus();
		void request_autofocus_stop();
		void request_autocontrast(WindowKind kind);
		void request_filter2D_roi_update();
		void request_filter2D_roi_end();
		void request_update_n(const unsigned short n);
		void request_update_unwrap_size(const unsigned size);
		void request_unwrapping_1d(const bool value);
		void request_unwrapping_2d(const bool value);
		void request_average(ConcurrentDeque<Tuple4f>* output);
		void request_average_stop();
		void request_average_record(ConcurrentDeque<Tuple4f>* output, const uint n);
		void request_termination();
		/*!
		 * \brief Updates the queues size
		 */
		void update_acc_parameter(
			std::unique_ptr<Queue>& gpu_img_acc,
			std::atomic<bool>& enabled,
			std::atomic<uint>& queue_length,
			camera::FrameDescriptor new_fd);

		/*! \brief Execute one iteration of the ICompute.
		*
		* * Checks the number of frames in input queue that must at least
		* nSize*.
		* * Call each function of the ICompute.
		* * Enqueue the output frame contained in gpu_output_buffer.
		* * Dequeue one frame of the input queue.
		* * Check if a ICompute refresh has been requested.
		*
		* The ICompute can not be interrupted for parameters changes until the
		* refresh method is called. */
		virtual void exec() = 0;

		void			create_stft_slice_queue();
		void			delete_stft_slice_queue();
		std::unique_ptr<Queue>& get_stft_slice_queue(int i);
		bool			get_cuts_request();
		bool			get_cuts_delete_request();
		bool			get_request_refresh();
		void			set_gpib_interface(std::shared_ptr<gpib::IVisaInterface> gpib_interface);

		bool get_unwrap_1d_request()		const { return unwrap_1d_requested_; }
		bool get_unwrap_2d_request()		const { return unwrap_2d_requested_; }
		bool get_autofocus_request()		const { return autofocus_requested_; }
		bool get_autofocus_stop_request()	const { return autofocus_stop_requested_; }
		bool get_autocontrast_request()		const { return autocontrast_requested_; }
		bool get_autocontrast_slice_xz_request()		const { return autocontrast_slice_xz_requested_; }
		bool get_autocontrast_slice_yz_request()		const { return autocontrast_slice_yz_requested_; }
		bool get_refresh_request()			const { return refresh_requested_; }
		bool get_update_n_request()			const { return update_n_requested_; }
		bool get_stft_update_roi_request()	const { return stft_update_roi_requested_; }
		bool get_average_request()			const { return average_requested_; }
		bool get_average_record_request()	const { return average_record_requested_; }
		bool get_abort_construct_request()	const { return abort_construct_requested_; }
		bool get_termination_request()		const { return termination_requested_; }
		bool get_update_acc_request()		const { return update_acc_requested_; }
		bool get_update_ref_diff_request()	const { return update_ref_diff_requested_; }
		bool get_request_stft_cuts()		const { return request_stft_cuts_; }
		bool get_request_delete_stft_cuts() const { return request_delete_stft_cuts_; }

		void set_stft_frame_counter(uint value)
		{
			stft_env_.stft_frame_counter_ = value;
		}

		virtual std::unique_ptr<Queue>&	get_lens_queue() = 0;
		virtual std::unique_ptr<Queue>&	get_raw_queue() = 0;
	protected:

		virtual void refresh() = 0;
		virtual void allocation_failed(const int& err_count, std::exception& e);
		virtual bool update_n_parameter(unsigned short n);
		void request_queues();

		void fps_count();

		ICompute& operator=(const ICompute&) = delete;
		ICompute(const ICompute&) = delete;

	protected:
		/** Compute Descriptor. */
		ComputeDescriptor&	compute_desc_;

		/** Reference on the input queue, owned by MainWindow. */
		Queue&	input_;
		/** Reference on the output queue, owned by MainWindow. */
		Queue&	output_;

		/** Interface allowing to use the GPIB dll. */
		std::shared_ptr<gpib::IVisaInterface>	gpib_interface_;

		/** Main buffers. */
		CoreBuffers		buffers_;
		/** STFT environment. */
		Stft_env		stft_env_;
		/** Average environment. */
		Average_env		average_env_;
		/** Pland 2D. Used for spatial fft performed on the complex input frame. */
		cuda_tools::CufftHandle	plan2d_;

		/** Chrono counting time between two iteration (Taking into account steps, since it is executing at the end of pipe). */
		std::chrono::time_point<std::chrono::steady_clock>	past_time_;

		/** Counting pipe iteration, in order to update fps only every 100 iterations. */
		uint	frame_count_;

		/** YZ Image Accumulation Queue. */
		std::unique_ptr<Queue>	gpu_img_acc_yz_;
		/** XZ Image Accumulation Queue. */
		std::unique_ptr<Queue>	gpu_img_acc_xz_;

		std::atomic<bool>	unwrap_1d_requested_;
		std::atomic<bool>	unwrap_2d_requested_;
		std::atomic<bool>	autofocus_requested_;
		std::atomic<bool>	autofocus_stop_requested_;
		std::atomic<bool>	autocontrast_requested_;
		std::atomic<bool>	autocontrast_slice_xz_requested_;
		std::atomic<bool>	autocontrast_slice_yz_requested_;
		std::atomic<bool>	refresh_requested_;
		std::atomic<bool>	update_n_requested_;
		std::atomic<bool>	stft_update_roi_requested_;
		std::atomic<bool>	average_requested_;
		std::atomic<bool>	average_record_requested_;
		std::atomic<bool>	abort_construct_requested_;
		std::atomic<bool>	termination_requested_;
		std::atomic<bool>	update_acc_requested_;
		std::atomic<bool>	update_ref_diff_requested_;
		std::atomic<bool>	request_stft_cuts_;
		std::atomic<bool>	request_delete_stft_cuts_;
	};
}
