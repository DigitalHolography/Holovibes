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

 Implementation of FFT1, FFT2 and STFT algorithms. */
#pragma once

#include "pipeline_utils.hh"
#include "frame_desc.hh"
#include "rect.hh"
#include "cuda_tools\unique_ptr.hh"
#include "autofocus.hh"

namespace holovibes
{
	class ComputeDescriptor;
	struct Stft_env;
	struct CoreBuffers;

	namespace compute
	{
		class FourierTransform
		{
		public:
			FourierTransform(FnVector& fn_vect,
				const CoreBuffers& buffers,
				const std::unique_ptr<Autofocus>& autofocus,
				const camera::FrameDescriptor& fd,
				holovibes::ComputeDescriptor& cd,
				const cufftHandle& plan2d,
				Stft_env& stft_env);

			/*! \brief Enqueue the appropriate functions
			**
			** Should be called just after gpu_float_buffer is computed
			*/
			void allocate(unsigned int n);
			void insert_fft();
			void insert_stft();
			Queue* get_lens_queue();
		private:
			void insert_filter2d();
			void insert_fft1();
			void insert_fft2();
			void stft_handler();
			void enqueue_lens();

			//! Roi zone of Filter 2D
			units::RectFd					filter2d_zone_;

			//! Lens used for fresnel transform (During FFT1 and FFT2)
			cuda_tools::UniquePtr<cufftComplex> gpu_lens_;
			//! Lens Queue. Used for displaying the lens.
			std::unique_ptr<Queue>				gpu_lens_queue_;
			//! Filter 2D buffer. Contains one frame.
			cuda_tools::UniquePtr<cufftComplex>	gpu_filter2d_buffer_;
			//! Crop STFT buffer. Contains nsamples frames. Used to apply STFT on smaller areas than the whole window.
			cuda_tools::UniquePtr<cufftComplex> gpu_cropped_stft_buf_;

			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;
			//! Main buffers
			const CoreBuffers&				buffers_;
			//! Autofocus feature. Used to retrieve the correct zindex to compute.
			const std::unique_ptr<Autofocus>& autofocus_;
			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;
			//! Compute Descriptor
			ComputeDescriptor&				cd_;
			//! Pland 2D. Used by STFT.
			const cufftHandle&				plan2d_;
			//! STFT environment.
			Stft_env&						stft_env_;
		};
	}
}