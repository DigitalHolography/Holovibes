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

# include "icompute.hh"

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
		Pipe(Queue& input, Queue& output, ComputeDescriptor& desc);
		virtual ~Pipe();

	protected:

		/*! \brief Execute one processing iteration.
		*
		* * Checks the number of frames in input queue, that must at least
		* be nsamples.
		* * Call each function stored in the FnVector.
		* * Enqueue the output frame contained in gpu_output_buffer.
		* * Dequeue one frame of the input queue.
		* * Check if a ICompute refresh has been requested.
		*
		* The ICompute can not be interrupted for parameters changes until the
		* refresh method is called. */

		void			direct_refresh();
		virtual void	refresh();
		void			*get_enqueue_buffer();
		virtual void	exec();
		virtual bool	update_n_parameter(unsigned short n);
		void			request_queues();
		void			autofocus_caller(float* input, cudaStream_t stream) override;

	private:
		FnVector		fn_vect_;

		cufftComplex	*gpu_input_buffer_;
		void			*gpu_output_buffer_;
		cufftComplex	*gpu_input_frame_ptr_;
	};
}