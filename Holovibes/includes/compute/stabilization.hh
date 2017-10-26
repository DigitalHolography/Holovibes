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
# include "cuda_tools/unique_ptr.hh"
# include "cuda_tools/array.hh"
# include "pipeline_utils.hh"
# include "frame_desc.hh"
# include "queue.hh"
# include "Rectangle.hh"

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
				float* const& gpu_float_buffer,
				const camera::FrameDescriptor& fd,
				const holovibes::ComputeDescriptor& cd);

			/*! \brief Enqueue the appropriate functions
			**
			** Should be called just after gpu_float_buffer is computed
			*/
			void insert_post_img_type();

		private:

			void insert_average_compute();
			void insert_correlation();
			void insert_extremums();
			void insert_stabilization();
			void insert_float_buffer_overwrite();

			void compute_correlation(const float *x, const float *y);
			void compute_convolution(const float* x, const float* y, float* out);
			gui::Rectangle get_squared_zone() const;

			/// Buffer to keep the convolution product
			cuda_tools::Array<float>				convolution_;

			/// Buffer used to temporaly store the average, to compare it with current frame
			cuda_tools::UniquePtr<float>				float_buffer_average_;

			/// Current image shift
			/// {
			int							shift_x;
			int							shift_y;
			/// }

			std::unique_ptr<Queue>			accumulation_queue_;

			/// Pipe data
			/// {
			/// Vector function in which we insert the processing
			FnVector& fn_vect_;
			/// The whole image for this frame
			float* const& gpu_float_buffer_;
			/// Describes the frame size
			const camera::FrameDescriptor& fd_;

			const ComputeDescriptor& cd_;
			/// }
		};
	}
}