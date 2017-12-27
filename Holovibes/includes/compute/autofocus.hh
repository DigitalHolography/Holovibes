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

Implementation of autofocus feature. */
#pragma once

#include "pipeline_utils.hh"
#include "rect.hh"
#include "frame_desc.hh"
#include "cuda_tools\unique_ptr.hh"

namespace holovibes
{
	class ComputeDescriptor;
	class ICompute;
	struct CoreBuffers;
	/*! \brief Contains all functions and structure for computations variables */
	namespace compute
	{
		/*! \brief Describing the autofocus state */
		enum af_state
		{
			STOPPED, /**< Autofocus not requested */
			COPYING, /**< Copying frames when autofocus is used with stft */
			RUNNING /**< Autofocus running after everything initialized correctly*/
		};

		/*! \brief Discribing the autofocus environment */
		struct af_env
		{
			float				z; /**< intern z used for autofocus computation */
			float				z_min; /**< minimal z for each loop */
			float				z_max; /**< maximal z for each loop */

			float				z_step; /**< value incrementing z at each iteration */
			unsigned int		z_iter; /**< number of loops remaining */
			float				af_z; /**< best z found during last loop */

			std::vector<float>	focus_metric_values; /**< vector containing the values given by the evaluating function*/

			units::RectFd		zone; /**< zone where autofocus is applied */
			unsigned int		af_square_size; /**< size of the square zone where autofocus is applied */
			cuda_tools::UniquePtr<float> gpu_float_buffer_af_zone; /**< zone of gpu_float_buffer_ where autocus is applied */

			size_t				gpu_input_size; /**< size of gpu_input_buffer_tmp */
			size_t				gpu_frame_size; /**< size of one frame inside gpu_input_buffer_tmp */
			cuda_tools::UniquePtr<cufftComplex>	gpu_input_buffer_tmp; /**< buffer saving the frames to work on the same images at each iteration. It contains #img when stft is disabled, and an hardcoded number when enabled. */

			unsigned short		nSize; /**< hardcoded value of frames to save when stft is enabled. Must be grater than 2. */
			unsigned short		p; /**< hardcoded value of p when stft is enabled */

			unsigned short		old_nSize; /**< old value of nSize */
			unsigned short		old_p; /**< old value of p*/

			unsigned int		old_steps; /**< old value of stft steps*/
			int					stft_index; /**< current index of the frame to save/copy when stft is enabled. We need it since the input_length_ is equal to 1 when stft */
			enum af_state		state; /**< state of autofocus process */
		};

		class Autofocus
		{
		public:
			/** \brief Constructor.
			
			*/
			Autofocus(FnVector& fn_vect,
				const CoreBuffers& buffers,
				holovibes::Queue& input,
				holovibes::ComputeDescriptor& cd,
				ICompute *Ic);

			/** \brief see autofocus_init()
			
			*/
			void insert_init();
			/** \brief see autofocus_restore()
			
			*/
			void insert_restore();
			/** \brief see autofocus_caller()
			
			*/
			void insert_autofocus();
			/** \brief insert the copy of the input buffer into a tmp buffer
			
			*/
			void insert_copy();

			/** \brief Get the zindex used for the spatial fft.
			
			*/
			float get_zvalue() const;

			/** \brief Get the autofocus state.
			
			*/
			af_state get_state() const
			{
				return af_env_.state;
			}

		private:

			/*! \brief restores the input frames saved in the gpu_input_buffer_
			* \param input_buffer Destination buffer (gpu_input_buffer_) */
			void autofocus_restore();

			/*! \brief Initialize the structure af_env_
			
			*/
			void autofocus_init();

			/*! \brief This is the main part of the autofocus. It will copy
			* the square area where autofocus is applied, call the evaluating function, and updates every value for next iteration.
			* \param input buffer of float containing the image where autofocus is applied (gpu_float_buffer_) */
			void autofocus_caller(cudaStream_t stream = 0);

			/*! \brief Resetting the structure af_env_ for next use
			
			*/
			void autofocus_reset();

			//! Containing every special vairables needed to run autofocus
			af_env	af_env_;

			/// Main pipe buffers.
			const CoreBuffers&				buffers_;
			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;
			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;
			/// Variables needed for the computation in the pipe
			ComputeDescriptor&				cd_;
			/// input queue containing raw images (on GPU). Used to fill tmp buffer in the autofocus initialization
			Queue&							input_;
			/// Pointer on the pipe.
			ICompute*						Ic_;
		};
	}
}
