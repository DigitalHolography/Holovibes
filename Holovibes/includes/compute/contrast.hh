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

   Implementation of the rendering features. */
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
	struct Stft_env;
	enum WindowKind;

	namespace compute
	{
		using uint = unsigned int;

		class Contrast
		{
		public:
			/** \brief Constructor.
			
			*/
			Contrast(FnVector& fn_vect,
				const CoreBuffers& buffers,
				Average_env& average_env,
				ComputeDescriptor& cd,
				const camera::FrameDescriptor& input_fd,
				const camera::FrameDescriptor& output_fd,
				ICompute* Ic);

			/** \brief insert the functions relative to the fft shift.

			*/
			void insert_fft_shift();
			/** \brief insert the functions relative to noise and signal average.

			 */
			void insert_average(std::atomic<bool>& record_request);
			/** \brief insert the functions relative to the log10.

			*/
			void insert_log();
			/** \brief insert the functions relative to the contrast.

			*/
			void insert_contrast(std::atomic<bool>& autocontrast_request, std::atomic<bool>& autocontrast_slice_xz_request, std::atomic<bool>& autocontrast_slice_yz_request);

		private:
			/** \brief insert the average computation.

			*/
			void insert_main_average();
			/** \brief insert the average recording.

			*/
			void insert_average_record();

			/** \brief insert the log10 on the XY window

			*/
			void insert_main_log();
			/** \brief insert the log10 on the slices

			*/
			void insert_slice_log();

			/** \brief insert the autocontrast

			*/
			void insert_autocontrast(std::atomic<bool>& autocontrast_request, std::atomic<bool>& autocontrast_slice_xz_request, std::atomic<bool>& autocontrast_slice_yz_request);
			/** \brief insert the constrast on the XY window

			*/
			void insert_main_contrast();
			/** \brief insert the contrast on the slices

			*/
			void insert_slice_contrast();

			/** \brief Calls autocontrast and set the correct contrast variables
			
			*/
			void autocontrast_caller(float *input,
				const uint			size,
				const uint			offset,
				WindowKind			view,
				cudaStream_t		stream = 0);

			/*! \see request_average_record
			* \brief Call the average algorithm, store the result and count n
			* iterations. Request the ICompute to refresh when record is over.
			* \param signal Signal zone
			* \param noise Noise zone */
			void average_record_caller(
				const units::RectFd& signal,
				const units::RectFd& noise,
				cudaStream_t stream = 0);


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
			/// Pointer on the parent.
			ICompute*						Ic_;
		};
	}
}
