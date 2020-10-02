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

# include "cuda_tools/unique_ptr.hh"
# include "cuda_tools/array.hh"
# include "cuda_tools/nppi_data.hh"
# include "pipeline_utils.hh"
# include "frame_desc.hh"
# include "queue.hh"
# include "rect.hh"

namespace holovibes
{
	class Queue;
	class ComputeDescriptor;
	struct CoreBuffers;

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
				const CoreBuffers& buffers,
				const camera::FrameDescriptor& fd,
				const holovibes::ComputeDescriptor& cd);

			/*!
			** \brief Enqueue the image accumulation.
			** Should be called just after gpu_float_buffer is computed
			*/
			void insert_image_accumulation();

		private:
			/*!
			** \brief Insert the computation of the average of the float frame.
			*/
			void insert_average_compute();

			/*!
			** \brief Insert the copy of the corrected buffer into the float buffer.
			*/
			void insert_float_buffer_overwrite();

			/// Buffer used to temporaly store the average, to compare it with current frame
			cuda_tools::UniquePtr<float>	float_buffer_average_;

			/// Queue accumulating XY frames.
			std::unique_ptr<Queue>			accumulation_queue_;

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
