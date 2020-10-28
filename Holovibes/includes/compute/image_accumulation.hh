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

/*!
** \brief Contains functions relative to image accumulation.
*/

#pragma once

#include "cuda_tools/unique_ptr.hh"
#include "cuda_tools/array.hh"
#include "cuda_tools/nppi_data.hh"
#include "pipeline_utils.hh"
#include "frame_desc.hh"
#include "queue.hh"
#include "rect.hh"

namespace holovibes
{
	class Queue;
	class ComputeDescriptor;
	struct CoreBuffers;
	struct ImageAccEnv;

	/*! \brief Contains all functions and structure for computations variables */
	namespace compute
	{
		/*! \class ImageAccumulation
		**
		** Class that manages the image accumulation
		** It manages its own buffer, initialized when needed
		** It should be a member of the Pipe class
		*/
		class ImageAccumulation
		{
		public:
			/*!
			** \brief Constructor.
			*/
			ImageAccumulation(FnVector& fn_vect,
				ImageAccEnv& image_acc_env,
				const CoreBuffers& buffers,
				const camera::FrameDescriptor& fd,
				const holovibes::ComputeDescriptor& cd);

			/*!
			** \brief Enqueue the image accumulation.
			** Should be called just after gpu_float_buffer is computed
			*/
			void insert_image_accumulation();

			/*!
			** \brief Handle the allocation of the accumulation queues and average frames
			*/
			void allocate_accumulation_queues();

		private:
			/*!
			** \brief Compute average on one view
			*/
			void compute_average(
				std::unique_ptr<Queue>& gpu_accumulation_queue,
				float* gpu_input_frame,
				float* gpu_ouput_average_frame,
				const unsigned int image_acc_level,
				const size_t frame_res);

			/*!
			** \brief Insert the average computation of the float frame.
			*/
			void insert_compute_average();

			/*!
			** \brief Insert the copy of the corrected buffer into the float buffer.
			*/
			void insert_copy_accumulation_result();

			/*!
			** \brief Handle the allocation of a accumulation queue and average frame
			*/
			void allocate_accumulation_queue(
				std::unique_ptr<Queue>& gpu_accumulation_queue,
				cuda_tools::UniquePtr<float>& gpu_average_frame,
				const unsigned int accumulation_level,
				const camera::FrameDescriptor fd);

		private: /* Attributes */
			/// Image Accumulation environment
			ImageAccEnv& image_acc_env_;

			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;

			/// Main buffers
			const CoreBuffers&				buffers_;

			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;
			/// Compute Descriptor
			const ComputeDescriptor&		cd_;
		};
	}
}
