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
# include "gpib_controller.hh"
# include "frame_desc.hh"

namespace holovibes
{
# ifndef TUPLE4F
# define TUPLE4F
	using	Tuple4f = std::tuple<float, float, float, float>;
# endif
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

		struct af_env
		{
			float				z;
			float				z_min;
			float				z_max;
			float				z_step;
			unsigned int		z_iter;
			float				af_z;
			std::vector<float>	focus_metric_values;
			gui::Rectangle		zone;
			float				*gpu_float_buffer_af_zone;
			cufftComplex		*gpu_input_buffer_tmp;
			size_t				gpu_input_size;
			unsigned int		af_square_size;
			unsigned int		nsamples;
			unsigned int		p;
			unsigned int		old_nsamples;
			unsigned int		old_p;
		};

		enum ref_state
		{
			ENQUEUE,
			COMPUTE
		};

		void autofocus_init();

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
		void request_autocontrast();
		void request_filter2D_roi_update();
		void request_filter2D_roi_end();
		void request_update_n(const unsigned short n);
		void request_update_unwrap_size(const unsigned size);
		void request_unwrapping_1d(const bool value);
		void request_unwrapping_2d(const bool value);
		void request_average(ConcurrentDeque<Tuple4f>* output);
		void request_average_stop();
		void request_average_record(ConcurrentDeque<Tuple4f>* output, const uint n);
		void request_float_output(Queue* fqueue);
		void request_float_output_stop();
		void request_complex_output(Queue* fqueue);
		void request_complex_output_stop();
		void request_termination();
		void queue_enqueue(void* input, Queue* queue);
		void stft_handler(cufftComplex* input, cufftComplex* output);
		/*!
		 * \brief Updates the queues size
		 */
		void update_acc_parameter(
			Queue*& gpu_img_acc,
			std::atomic<bool>& enabled,
			std::atomic<uint>& queue_length,
			camera::FrameDescriptor new_fd);
		void update_ref_diff_parameter();

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
		void			create_3d_vision_queue();
		void			delete_3d_vision_queue();
		void			update_stft_slice_queue();
		Queue&			get_stft_slice_queue(int i);
		bool			get_cuts_request();
		bool			get_cuts_delete_request();
		bool			get_request_refresh();
		Queue&			get_3d_vision_queue();
		void			set_gpib_interface(std::shared_ptr<gpib::IVisaInterface> gpib_interface);

		bool get_unwrap_1d_request()		const { return (unwrap_1d_requested_.load()); }
		bool get_unwrap_2d_request()		const { return (unwrap_2d_requested_.load()); }
		bool get_autofocus_request()		const { return (autofocus_requested_.load()); }
		bool get_autofocus_stop_request()	const { return (autofocus_stop_requested_.load()); }
		bool get_autocontrast_request()		const { return (autocontrast_requested_.load()); }
		bool get_refresh_request()			const { return (refresh_requested_.load()); }
		bool get_update_n_request()			const { return (update_n_requested_.load()); }
		bool get_stft_update_roi_request()	const { return (stft_update_roi_requested_.load()); }
		bool get_average_request()			const { return (average_requested_.load()); }
		bool get_average_record_request()	const { return (average_record_requested_.load()); }
		bool get_float_output_request()		const { return (float_output_requested_.load()); }
		bool get_complex_output_request()	const { return (complex_output_requested_.load()); }
		bool get_abort_construct_request()	const { return (abort_construct_requested_.load()); }
		bool get_termination_request()		const { return (termination_requested_.load()); }
		bool get_update_acc_request()		const { return (update_acc_requested_.load()); }
		bool get_update_ref_diff_request()	const { return (update_ref_diff_requested_.load()); }
		bool get_request_stft_cuts()		const { return (request_stft_cuts_.load()); }
		bool get_request_delete_stft_cuts() const { return (request_delete_stft_cuts_.load()); }
		bool get_request_3d_vision()		const { return (request_3d_vision_.load()); }
		bool get_request_delete_3d_vision()	const { return (request_delete_3d_vision_.load()); }


	protected:

		virtual void refresh();
		virtual void allocation_failed(const int& err_count, std::exception& e);
		virtual bool update_n_parameter(unsigned short n);

		/*! \{ \name caller methods (helpers)
		*
		* For some features, it might be necessary to do special treatment. For
		* example, store a returned value in a std::vector. */

		static void autocontrast_caller(float				*input,
			const uint			size,
			const uint			offset,
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
		void average_stft_caller(cufftComplex* input,
			const unsigned int width,
			const unsigned int height,
			const unsigned int width_roi,
			const unsigned int height_roi,
			gui::Rectangle& signal_zone,
			gui::Rectangle& noise_zone,
			const unsigned int nsamples,
			cudaStream_t stream);

		virtual void autofocus_caller(float* input, cudaStream_t stream);
		void record_float(float* float_output, cudaStream_t stream);
		void record_complex(cufftComplex* complex_output, cudaStream_t stream);
		void handle_reference(cufftComplex* input, const unsigned int nframes);
		void handle_sliding_reference(cufftComplex* input, const unsigned int nframes);
		void fps_count();

		ICompute& operator=(const ICompute&) = delete;
		ICompute(const ICompute&) = delete;

	protected:
		ComputeDescriptor&	compute_desc_;

		Queue&	input_;
		Queue&	output_;

		std::shared_ptr<UnwrappingResources>	unwrap_res_;
		std::shared_ptr<UnwrappingResources_2d>	unwrap_res_2d_;
		std::shared_ptr<gpib::IVisaInterface>	gpib_interface_;

		std::mutex		stftGuard;
		void			*gpu_float_cut_xz_;
		void			*gpu_float_cut_yz_;
		void			*gpu_ushort_cut_xz_;
		void			*gpu_ushort_cut_yz_;
		float			*gpu_float_buffer_;
		cufftComplex	*gpu_stft_buffer_;
		cufftComplex	*gpu_tmp_input_;
		cufftComplex	*gpu_special_queue_;
		cufftComplex	*gpu_lens_;
		cufftHandle		plan3d_;
		cufftHandle		plan2d_;
		cufftHandle		plan1d_;
		cufftHandle		plan1d_stft_;
		float			*gpu_kernel_buffer_;
		uint			gpu_special_queue_start_index;
		uint			gpu_special_queue_max_index;

		uint	input_length_;
		Queue	*fqueue_;
		uint	curr_elt_stft_;

		ConcurrentDeque<Tuple4f>* average_output_;
		std::chrono::time_point<std::chrono::steady_clock>	past_time_;

		uint	average_n_;
		uint	frame_count_;
		af_env	af_env_;

		Queue	*gpu_img_acc_xy_;
		Queue	*gpu_img_acc_yz_;
		Queue	*gpu_img_acc_xz_;
		Queue	*gpu_stft_queue_;
		Queue	*gpu_3d_vision;
		std::unique_ptr<Queue>	gpu_stft_slice_queue_xz;
		std::unique_ptr<Queue>	gpu_stft_slice_queue_yz;
		Queue	*gpu_ref_diff_queue_;

		enum ref_state	ref_diff_state_;
		cufftComplex	*gpu_filter2d_buffer;
		uint			ref_diff_counter;
		uint			stft_frame_counter;
		bool			stft_handle;

		std::atomic<bool>	unwrap_1d_requested_;
		std::atomic<bool>	unwrap_2d_requested_;
		std::atomic<bool>	autofocus_requested_;
		std::atomic<bool>	autofocus_stop_requested_;
		std::atomic<bool>	autocontrast_requested_;
		std::atomic<bool>	refresh_requested_;
		std::atomic<bool>	update_n_requested_;
		std::atomic<bool>	stft_update_roi_requested_;
		std::atomic<bool>	average_requested_;
		std::atomic<bool>	average_record_requested_;
		std::atomic<bool>	float_output_requested_;
		std::atomic<bool>	complex_output_requested_;
		std::atomic<bool>	abort_construct_requested_;
		std::atomic<bool>	termination_requested_;
		std::atomic<bool>	update_acc_requested_;
		std::atomic<bool>	update_ref_diff_requested_;
		std::atomic<bool>	request_stft_cuts_;
		std::atomic<bool>	request_delete_stft_cuts_;
		std::atomic<bool>	request_3d_vision_;
		std::atomic<bool>	request_delete_3d_vision_;
	};
}
