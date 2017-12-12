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
	struct CoreBuffers
	{
		// input buffer
		cuda_tools::UniquePtr<cufftComplex>	gpu_input_buffer_ = nullptr;

		// float buffers
		cuda_tools::UniquePtr<float>		gpu_float_buffer_ = nullptr;
		uint								gpu_float_buffer_size_ = 0;
		cuda_tools::UniquePtr<void>			gpu_float_cut_xz_ = nullptr;
		cuda_tools::UniquePtr<void>			gpu_float_cut_yz_ = nullptr;

		// Output buffers
		cuda_tools::UniquePtr<void>			gpu_output_buffer_ = nullptr;
		cuda_tools::UniquePtr<void>			gpu_ushort_cut_xz_ = nullptr;
		cuda_tools::UniquePtr<void>			gpu_ushort_cut_yz_ = nullptr;
	};

	struct Stft_env
	{
		// Lock
		std::mutex							stftGuard_;

		// Buffers
		cuda_tools::UniquePtr<cufftComplex>	gpu_stft_buffer_ = nullptr;
		std::unique_ptr<Queue>				gpu_stft_queue_ = nullptr;
		std::unique_ptr<Queue>				gpu_stft_slice_queue_xz = nullptr;
		std::unique_ptr<Queue>				gpu_stft_slice_queue_yz = nullptr;
		// Plan
		cuda_tools::CufftHandle				plan1d_stft_;

		// Handling steps
		bool								stft_handle_ = false;
		uint								stft_frame_counter_ = 1;
	};

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
			camera::FrameDescriptor new_fd,
			float depth = 4.f);

		/*! \brief Return true while ICompute is recording float. */

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

		void			create_stft_slice_queue();
		void			delete_stft_slice_queue();
		void			update_stft_slice_queue();
		Queue&			get_stft_slice_queue(int i);
		bool			get_cuts_request();
		bool			get_cuts_delete_request();
		bool			get_request_refresh();
		void			set_gpib_interface(std::shared_ptr<gpib::IVisaInterface> gpib_interface);

		bool get_unwrap_1d_request()		const { return unwrap_1d_requested_; }
		bool get_unwrap_2d_request()		const { return unwrap_2d_requested_; }
		bool get_autofocus_request()		const { return autofocus_requested_; }
		bool get_autofocus_stop_request()	const { return autofocus_stop_requested_; }
		bool get_autocontrast_request()		const { return autocontrast_requested_; }
		bool get_autocontrast_slice_request()		const { return autocontrast_slice_requested_; }
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

		virtual Queue*	get_lens_queue();


	protected:

		virtual void refresh() = 0;
		virtual void allocation_failed(const int& err_count, std::exception& e);
		virtual bool update_n_parameter(unsigned short n);

		void record(void* output, cudaStream_t stream = 0);
		void fps_count();

		ICompute& operator=(const ICompute&) = delete;
		ICompute(const ICompute&) = delete;

	protected:
		ComputeDescriptor&	compute_desc_;

		Queue&	input_;
		Queue&	output_;

		std::shared_ptr<gpib::IVisaInterface>	gpib_interface_;

		CoreBuffers		buffers_;
		Stft_env		stft_env_;
		Average_env		average_env_;
		cuda_tools::CufftHandle	plan2d_;

		std::unique_ptr<Queue> fqueue_;

		std::chrono::time_point<std::chrono::steady_clock>	past_time_;

		uint	frame_count_;

		std::unique_ptr<Queue>	gpu_img_acc_yz_;
		std::unique_ptr<Queue>	gpu_img_acc_xz_;

		std::atomic<bool>	unwrap_1d_requested_;
		std::atomic<bool>	unwrap_2d_requested_;
		std::atomic<bool>	autofocus_requested_;
		std::atomic<bool>	autofocus_stop_requested_;
		std::atomic<bool>	autocontrast_requested_;
		std::atomic<bool>	autocontrast_slice_requested_;
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
