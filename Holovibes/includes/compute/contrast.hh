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
#include <atomic>

#include "frame_desc.hh"
#include "pipeline_utils.hh"
#include "queue.hh"

namespace holovibes
{
	class ComputeDescriptor;
	namespace compute
	{
		using uint = unsigned int;

		class Contrast
		{
		public:
			Contrast(FnVector& fn_vect,
				float* const& gpu_float_buffer,
				const uint& gpu_float_buffer_size,
				void* const& gpu_float_cut_xz,
				void* const& gpu_float_cut_yz,
				ComputeDescriptor& cd,
				const camera::FrameDescriptor& output_fd,
				Queue*& gpu_3d_vision,
				std::atomic<bool>& request);

			void insert_contrast();

		private:
			void insert_autocontrast();
			void insert_vision3d_contrast();
			void insert_main_contrast();
			void insert_slice_contrast();

			void autocontrast_caller(float *input,
						const uint			size,
						const uint			offset,
						cudaStream_t		stream = 0);

			/// The whole image for this frame
			float* const&					gpu_float_buffer_;
			/// Size of the gpu_float_buffer (size of a frame)
			const uint&						gpu_float_buffer_size_;
			/// The whole image for this frame in XZ
			void* const&					gpu_float_cut_xz_;
			/// The whole image for this frame in YZ
			void* const&					gpu_float_cut_yz_;

			/// Pipe data
			/// {
			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;
			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;
			/// Variables needed for the computation in the pipe
			ComputeDescriptor&				cd_;
			/// output queue for 3d vision mode
			Queue*&							gpu_3d_vision_;
			/// Autocontrast request
			std::atomic<bool>&				request_;
			/// }
		};
	}
}
