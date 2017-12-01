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

#include "frame_desc.hh"
#include "pipeline_utils.hh"
#include "queue.hh"
#include "rect.hh"

namespace holovibes
{
	class ComputeDescriptor;
	class ICompute;
	struct CoreBuffers;
	struct Average_env;
	enum WindowKind;

	namespace compute
	{
		using uint = unsigned int;

		class Contrast
		{
		public:
			Contrast(FnVector& fn_vect,
				const CoreBuffers& buffers,
				Average_env& average_env,
				ComputeDescriptor& cd,
				const camera::FrameDescriptor& input_fd,
				const camera::FrameDescriptor& output_fd,
				const std::unique_ptr<Queue>& gpu_3d_vision,
				ICompute* Ic);

			void insert_fft_shift();
			void insert_average(std::atomic<bool>& record_request);
			void insert_log();
			void insert_contrast(std::atomic<bool>& autocontrast_request);

		private:
			void insert_main_average();
			void insert_average_record();

			void insert_main_log();
			void insert_slice_log();

			void insert_autocontrast(std::atomic<bool>& autocontrast_request);
			void insert_vision3d_contrast();
			void insert_main_contrast();
			void insert_slice_contrast();

			void autocontrast_caller(float *input,
				const uint			size,
				const uint			offset,
				WindowKind			view,
				cudaStream_t		stream = 0);

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
				const units::RectFd& signal,
				const units::RectFd& noise,
				cudaStream_t stream = 0);


			/// Pipe data
			/// {
			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;
			/// Main buffers
			const CoreBuffers&				buffers_;
			/// Average variables
			Average_env&					average_env_;
			/// Describes the input frame size
			const camera::FrameDescriptor& input_fd_;
			/// Describes the output frame size
			const camera::FrameDescriptor&	fd_;
			/// Variables needed for the computation in the pipe
			ComputeDescriptor&				cd_;
			/// output queue for 3d vision mode
			const std::unique_ptr<Queue>&	gpu_3d_vision_;
			///
			ICompute*						Ic_;
			/// }
		};
	}
}
