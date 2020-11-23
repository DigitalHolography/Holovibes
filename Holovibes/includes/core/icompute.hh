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

#include <atomic>
#include <memory>
#include <optional>

#include "config.hh"
#include "rect.hh"
#include "observable.hh"
#include "gpib_controller.hh"
#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "cufft_handle.hh"
#include "chart_point.hh"
#include "concurrent_deque.hh"

namespace holovibes
{
	class Queue;
	class ComputeDescriptor;
	enum WindowKind;
}

namespace holovibes
{
	/*! \brief Struct containing main buffers used by the pipe. */
	struct CoreBuffersEnv
	{
		/** Input buffer. Contains only one frame. We fill it with the input frame*/
		cuda_tools::UniquePtr<cufftComplex>		gpu_spatial_transformation_buffer = nullptr;

		/** Float buffer. Contains only one frame. We fill it with the correct computed p frame converted to float. */
		cuda_tools::UniquePtr<float>			gpu_postprocess_frame = nullptr;
		/** Size in components (size in byte / sizeof(float)) of the gpu_postprocess_frame.
		 Could be removed by changing gpu_postprocess_frame type to cuda_tools::Array. */
		unsigned int							gpu_postprocess_frame_size = 0;
		/** Float XZ buffer. Contains only one frame. We fill it with the correct computed p XZ frame. */
		cuda_tools::UniquePtr<float>			gpu_postprocess_frame_xz = nullptr;
		/** Float YZ buffer. Contains only one frame. We fill it with the correct computed p YZ frame. */
		cuda_tools::UniquePtr<float>			gpu_postprocess_frame_yz = nullptr;

		/** Unsigned Short output buffer. Contains only one frame, inserted after all postprocessing on float_buffer */
		cuda_tools::UniquePtr<unsigned short>	gpu_output_frame = nullptr;
		/** Unsigned Short XZ output buffer. Contains only one frame, inserted after all postprocessing on float_buffer_cut_xz */
		cuda_tools::UniquePtr<unsigned short>	gpu_output_frame_xz = nullptr;
		/** Unsigned Short YZ output buffer. Contains only one frame, inserted after all postprocessing on float_buffer_cut_yz */
		cuda_tools::UniquePtr<unsigned short>	gpu_output_frame_yz = nullptr;

		/** Contains only one frame used only for convolution */
		cuda_tools::UniquePtr<float>			gpu_convolution_buffer = nullptr;
	};

	/*! \brief Struct containing variables related to the batch in the pipe */
	struct BatchEnv
	{
		/*! \brief Current frames processed in the batch
		**
		** At index 0, batch_size frames are enqueued, spatial transformation is also executed in batch
		** Batch size frames are enqueued in the gpu_time_transformation_queue
		** This is done for perfomances reasons
		**
		** The variable is incremented unil it reachs batch_size in enqueue_multiple, then it is set back to 0
		*/
		uint batch_index = 0;
	};

	/*! \brief Struct containing variables related to STFT shared by multiple features of the pipe. */
	struct TimeTransformationEnv
	{
		/** STFT Queue. Constains time_transformation_size frames. It accumulates input frames after spatial fft,
		 in order to apply STFT only when the frame counter is equal to time_transformation_stride. */
		std::unique_ptr<Queue>				gpu_time_transformation_queue = nullptr;
		/** STFT buffer. Contains time_transformation_size frames. Contains the result of the STFT done on the STFT queue. */
		cuda_tools::UniquePtr<cufftComplex>	gpu_p_acc_buffer = nullptr;
		/** STFT XZ Queue. Contains the ouput of the STFT on slice XZ. Enqueued with gpu_float_buffer or gpu_ushort_buffer. */
		std::unique_ptr<Queue>				gpu_output_queue_xz = nullptr;
		/** STFT YZ Queue. Contains the ouput of the STFT on slice YZ. Enqueued with gpu_float_buffer or gpu_ushort_buffer. */
		std::unique_ptr<Queue>				gpu_output_queue_yz = nullptr;
		/** Plan 1D used for the STFT. */
		cuda_tools::CufftHandle				stft_plan;

		/** Hold the P frame after the time transformation computation. **/
		cuda_tools::UniquePtr<cufftComplex> gpu_p_frame;

		// The following are used for the PCA time transformation
		cuda_tools::UniquePtr<cuComplex> 	pca_cov = nullptr;
		cuda_tools::UniquePtr<cuComplex> 	pca_tmp_buffer = nullptr;
		cuda_tools::UniquePtr<float>		pca_eigen_values = nullptr;
		cuda_tools::UniquePtr<int>			pca_dev_info = nullptr;
	};

	/** \brief Structure containing variables related to the chart display and recording. */
	struct ChartEnv
	{
		std::unique_ptr<ConcurrentDeque<ChartPoint>> chart_display_queue_ = nullptr;
		std::unique_ptr<ConcurrentDeque<ChartPoint>> chart_record_queue_ = nullptr;
		unsigned int nb_chart_points_to_record_ = 0;
	};

	struct ImageAccEnv
	{
			/// Frame to temporaly store the average on XY view
			cuda_tools::UniquePtr<float>	gpu_float_average_xy_frame = nullptr;
			/// Queue accumulating the XY computed frames.
			std::unique_ptr<Queue>			gpu_accumulation_xy_queue = nullptr;

			/// Frame to temporaly store the average on XZ view
			cuda_tools::UniquePtr<float>	gpu_float_average_xz_frame = nullptr;
			/// Queue accumulating the XZ computed frames.
			std::unique_ptr<Queue>			gpu_accumulation_xz_queue = nullptr;

			/// Frame to temporaly store the average on YZ axis
			cuda_tools::UniquePtr<float>	gpu_float_average_yz_frame = nullptr;
			/// Queue accumulating the YZ computed frames.
			std::unique_ptr<Queue>			gpu_accumulation_yz_queue = nullptr;
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
			ComputeDescriptor& cd);
		virtual ~ICompute();

		void request_refresh();
		void request_output_resize(unsigned int new_output_size);
		void request_autocontrast(WindowKind kind);
		void request_filter2D_roi_update();
		void request_filter2D_roi_end();
		void request_update_time_transformation_size();
		void request_update_unwrap_size(const unsigned size);
		void request_unwrapping_1d(const bool value);
		void request_unwrapping_2d(const bool value);
		void request_display_chart();
		void request_disable_display_chart();
		void request_record_chart(unsigned int nb_chart_points_to_record);
		void request_disable_record_chart();
		void request_termination();
		void request_update_batch_size();
		void request_update_time_transformation_stride();
		void request_kill_raw_queue();
		void request_disable_lens_view();
		void request_allocate_raw_queue();

		/*! \brief Execute one iteration of the ICompute.
		*
		* * Checks the number of frames in input queue that must at least
		* time_transformation_size*.
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

		bool get_unwrap_1d_request()					const { return unwrap_1d_requested_; }
		bool get_unwrap_2d_request()					const { return unwrap_2d_requested_; }
		bool get_autocontrast_request()					const { return autocontrast_requested_; }
		bool get_autocontrast_slice_xz_request()		const { return autocontrast_slice_xz_requested_; }
		bool get_autocontrast_slice_yz_request()		const { return autocontrast_slice_yz_requested_; }
		bool get_refresh_request()						const { return refresh_requested_; }
		bool get_update_time_transformation_size_request()		const { return update_time_transformation_size_requested_; }
		bool get_stft_update_roi_request()				const { return stft_update_roi_requested_; }
		bool get_termination_request()					const { return termination_requested_; }
		bool get_request_time_transformation_cuts()				const { return request_time_transformation_cuts_; }
		bool get_request_delete_time_transformation_cuts() 		const { return request_delete_time_transformation_cuts_; }
		std::optional<unsigned int> get_output_resize_request() const { return output_resize_requested_; }
		bool get_kill_raw_queue_requested() 			const { return kill_raw_queue_requested_;}
		bool get_request_allocate_raw_queue() 			const { return request_allocate_raw_queue_;}
		bool get_chart_display_requested()				const { return chart_display_requested_; }
		std::optional<unsigned int> get_chart_record_requested() const { return chart_record_requested_; }
		bool get_disable_chart_display_requested()		const { return disable_chart_display_requested_; }
		bool get_disable_chart_record_requested()		const { return disable_chart_record_requested_; }

		virtual std::unique_ptr<Queue>&	get_lens_queue() = 0;

		/*! \brief Get the raw queue. Make allocation if needed */
		virtual std::unique_ptr<Queue>&	get_raw_queue();

		virtual std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_display_queue();

		virtual std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_record_queue();

	protected:

		virtual void refresh() = 0;
		virtual void pipe_error(const int& err_count, std::exception& e);
		virtual bool update_time_transformation_size(const unsigned short time_transformation_size);

		/* Manage ressources */
		virtual void update_spatial_transformation_parameters();
		void init_cuts();
		void dispose_cuts();

		void fps_count();

		ICompute& operator=(const ICompute&) = delete;
		ICompute(const ICompute&) = delete;

	protected:
		/** Compute Descriptor. */
		ComputeDescriptor&	cd_;

		/** Reference on the input queue, owned by MainWindow. */
		Queue&	gpu_input_queue_;
		/** Reference on the output queue, owned by MainWindow. */
		Queue&	gpu_output_queue_;

		/** Interface allowing to use the GPIB dll. */
		std::shared_ptr<gpib::IVisaInterface>	gpib_interface_;

		/** Main buffers. */
		CoreBuffersEnv	buffers_;

		/** Batch environment */
		BatchEnv batch_env_;

		/** STFT environment. */
		TimeTransformationEnv time_transformation_env_;

		/** Chart environment. */
		ChartEnv chart_env_;

		/** Image accumulation environment */
		ImageAccEnv	image_acc_env_;

		/*! \brief Queue storing raw frames used by raw view and raw recording */
		std::unique_ptr<Queue> gpu_raw_queue_{ nullptr };

		/** Pland 2D. Used for spatial fft performed on the complex input frame. */
		cuda_tools::CufftHandle	spatial_transformation_plan_;

		/** Pland 2D. Used for unwrap 2D. */
		cuda_tools::CufftHandle	plan_unwrap_2d_;

		/** Chrono counting time between two iteration (Taking into account steps, since it is executing at the end of pipe). */
		std::chrono::time_point<std::chrono::steady_clock>	past_time_;

		/** Counting pipe iteration, in order to update fps only every 100 iterations. */
		unsigned int frame_count_;

		std::atomic<bool> unwrap_1d_requested_{ false };
		std::atomic<bool> unwrap_2d_requested_{ false };
		std::atomic<bool> autocontrast_requested_{ false };
		std::atomic<bool> autocontrast_slice_xz_requested_{ false };
		std::atomic<bool> autocontrast_slice_yz_requested_{ false };
		std::atomic<bool> refresh_requested_{ false };
		std::atomic<bool> update_time_transformation_size_requested_{ false };
		std::atomic<bool> stft_update_roi_requested_{ false };
		std::atomic<bool> chart_display_requested_{ false };
		std::atomic<bool> disable_chart_display_requested_{ false };
		std::atomic<std::optional<unsigned int>> chart_record_requested_{ std::nullopt };
		std::atomic<bool> disable_chart_record_requested_{ false };
		std::atomic<std::optional<unsigned int>> output_resize_requested_{ std::nullopt };
		std::atomic<bool> kill_raw_queue_requested_{ false };
		std::atomic<bool> request_allocate_raw_queue_{ false };
		std::atomic<bool> termination_requested_{ false };
		std::atomic<bool> request_time_transformation_cuts_{ false };
		std::atomic<bool> request_delete_time_transformation_cuts_{ false };
		std::atomic<bool> request_update_batch_size_{ false };
		std::atomic<bool> request_update_time_transformation_stride_{ false };
		std::atomic<bool> request_disable_lens_view_{ false };
	};
}
