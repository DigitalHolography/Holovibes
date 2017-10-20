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

#pragma once


# include <cufft.h>
# include <qglobal.h>
# include <cuComplex.h>
# include "cuda_unique_ptr.hh"
# include "pipeline_utils.hh"
# include "frame_desc.hh"
#include "queue.hh"

namespace holovibes
{
	class ComputeDescriptor;
	namespace compute
	{

		/*! \class Stabilization
		**
		** Class that manages the stabilization of the image
		** It manages its own buffer, initialized when needed
		** It should be a member of the Pipe class
		*/
		class Stabilization
		{
		public:
			Stabilization(FnVector& fn_vect,
				cuComplex* const& gpu_complex_frame,
				float* const& gpu_float_buffer,
				const camera::FrameDescriptor& fd,
				const holovibes::ComputeDescriptor& cd);

			/*! \brief Enqueue the appropriate functions
			**
			** Should be called just after gpu_float_buffer is computed
			*/
			void insert_post_img_type();

		private:

			void insert_average();
			void insert_convolution();
			void insert_stabilization();
			void insert_extremums();

			void compute_convolution(const float* x, const float* y, float* out);

			/// Buffer to keep the last frame, will be replaced by an average
			CudaUniquePtr<float>			last_frame_;

			/// Buffer to keep the convolution product
			CudaUniquePtr<float>			convolution_;

			/// Buffer used to temporaly store the average, to compare it with current frame
			CudaUniquePtr<float>			float_buffer_average_;

			/// Current image shift
			/// {
			uint							shift_x;
			uint							shift_y;
			/// }

			std::unique_ptr<Queue>			accumulation_queue_;

			/// Pipe data
			/// {
			FnVector& fn_vect_;
			cuComplex* const& gpu_complex_frame_;
			float* const& gpu_float_buffer_;
			const camera::FrameDescriptor& fd_;
			const ComputeDescriptor& cd_;
			/// }
		};
	}
}