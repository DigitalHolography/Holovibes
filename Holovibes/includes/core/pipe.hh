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
 * The Pipe is a sequential computing model, storing procedures
 * in a single container. */
#pragma once

#include "cuda_tools/unique_ptr.hh"
#include "icompute.hh"
#include "image_accumulation.hh"
#include "fourier_transform.hh"
#include "rendering.hh"
#include "converts.hh"
#include "postprocessing.hh"
#include "function_vector.hh"

namespace holovibes
{
	/*! \brief Pipe is a class that applies processing on input frames.
	 *
	 * # Why doing this way ?
	 *
	 * The goal of the pipe is to build a vector filled with functions to
	 * apply on frames. This way it avoids to have a monolithic method plenty of
	 * if/else following what the user wants to do. In most cases, the processing
	 * remains the same at runtime, most jump conditions will always be the same.
	 *
	 * When the pipe is refreshed, the vector is updated with last user
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
	 * Also, some events such as autoconstrast will be executed only
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
		Pipe(Queue& input, Queue& output, ComputeDescriptor& desc);
		virtual ~Pipe();

		/*! \brief Get the lens queue to display it.

		*/
		std::unique_ptr<Queue>&			get_lens_queue() override;

		/*! \brief Runs a function after the current pipe iteration ends

		 */
		void insert_fn_end_vect(std::function<void()> function);
		/*! \brief Calls autocontrast on the *next* pipe iteration on the wanted view

		 */
		void autocontrast_end_pipe(WindowKind kind);

		/*! \brief Returns the class containing every functions relative to the FF1, FF2 and STFT algorithm.

		*/
		compute::FourierTransform * get_fourier_transforms();

	protected:

		/*! \brief Execute one processing iteration.
		*
		* * Checks the number of frames in input queue, that must at least
		* be 1.
		* * Call each function stored in the FunctionVector.
		* * Call each function stored in the end FunctionVector, then clears it
		* * Enqueue the output frame contained in gpu_output_buffer.
		* * Dequeue one frame of the input queue.
		* * Check if a ICompute refresh has been requested.
		*
		* The ICompute can not be interrupted for parameters changes until the
		* refresh method is called.
		* If Holovibes crash in a cuda function right after updating something on the GUI,
		* It probably means that the ComputeDescriptor has been updated before the end of the iteration.
		* The pipe uses the ComputeDescriptor, and is refresh only at the end of the iteration. You
		* **must** wait until the end of the refresh, or use the insert_fn_end_vect function to update the
		* ComputeDescriptor, otherwise the end of the current iteration will be wrong, and will maybe crash. */
		virtual void	exec();

		/*! \brief Enqueue the main FunctionVector according to the requests.

		*/
		virtual void	refresh();

		/*! \brief Make requests at the beginning of the refresh.
		* Make the allocation of buffers when it is requested.
		* \return return false if an allocation failed.
		*/
		virtual bool make_requests();

		/*!
		** \brief Transfer from gpu_input_buffer to gpu_stft_queue fro time filtering
		*/
		void insert_transfer_for_time_filter();

		/*!
		** \brief Wait that there are at least a batch of frames in input queue
		*/
		void insert_wait_frames();

		/*!
		** \brief Enqueue a batch of frames of input queue for raw view
		*/
		void insert_raw_view_enqueue();

		/*!
		** \brief Enqueue the input frame in the output queue in raw mode
		*/
		void insert_raw_enqueue_output();

		/*!
		** \brief Enqueue the output frame in the output queue in hologram mode
		*/
		void insert_hologram_enqueue_output();

		/*!
		** \brief Request the computation of a autocontrast if the contrast and
		** the contrast refresh is enabled
		*/
		void insert_request_autocontrast();

		/*!
		** \brief Reset the batch index if time_filter_stride has been reached
		*/
		void insert_reset_batch_index();

		/*!
		** \brief This method is used to synchronize the thread compute and
		** the thread recorder. In addition, it performs the transfer of frames
		** to the queue used for the recording. The queue is referenced in the
		** lambda.
		**
		** Mainwindow creates the thread recorder and call the record method.
		** In the record method, the thread recorder set
		** request_recorder_copy_frames to true to request the transfer of
		** frames, Then, this thread waits for the request to be completed by
		** the thread compute. Then, the thread compute transfers the number of
		** frames requested. Then, set the flag (copy_frames_done) to tell the
		** transfer is done. This allows the thread recorder to know that the
		** transfer is actually done.
		** Has to be called only if a recording is being done
		**
		** \param copy_function the technique used to copy frame(s), enqueue
		** or copy multiple...
		*/
		void copy_frames_for_recording(std::function<void()> copy_function);

		/*!
		** \brief Enqueue a frame in a output queue. The behavior is different
		** whether a recording is being done. Call the copy frames for recording
		** method if it is recording. Do a simple enqueue otherwise.
		**
		** \param output_queue Queue in which the frame is enqueued
		** \param frame Frame to enqueue
		** \param is_recording Flag to check if it is recording
		** \param error Error message when an error occurs
		*/
		void enqueue_output(Queue& output_queue,
							unsigned short* frame,
							bool is_recording,
							const std::string& error);

		/*!
		** \brief Enqueue a frame in an output queue
		**
		** \param output_queue Queue in which the frame is enqueued
		** \param frame Frame to enqueue
		** \param error Error message when an error occurs
		*/
		void safe_enqueue_output(Queue& output_queue,
									unsigned short* frame,
									const std::string& error);

	private:
		//! Vector of functions that will be executed in the exec() function.
		FunctionVector fn_compute_vect_;

		//! Vecor of functions that will be executed once, after the execution of fn_compute_vect_.
		FunctionVector fn_end_vect_;
		/*! Mutex that prevents the insertion of a function during its execution.
		    Since we can insert functions in fn_end_vect_ from other threads (MainWindow), we need to lock it.
		*/
		std::mutex		fn_end_vect_mutex_;

		std::unique_ptr<compute::ImageAccumulation> image_accumulation_;
		std::unique_ptr<compute::FourierTransform> fourier_transforms_;
		std::unique_ptr<compute::Rendering> rendering_;
		std::unique_ptr<compute::Converts> converts_;
		std::unique_ptr<compute::Postprocessing> postprocess_;


		/** Remaining number of raw frames to copy (on recording) */
		unsigned int remaining_raw_frames_copy_;

		/*! \brief Iterates and executes function of the pipe.

		  It will first iterate over fn_compute_vect_, then over function_end_pipe_. */
		void run_all();
	};
}
