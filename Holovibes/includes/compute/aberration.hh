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
 * Contains functions relative to aberration corrections. */
#pragma once

# include "cuda_tools/unique_ptr.hh"
# include "cuda_tools/array.hh"
# include "pipeline_utils.hh"
# include "frame_desc.hh"
# include "queue.hh"
# include "rect.hh"

namespace holovibes
{
	class ComputeDescriptor;
	struct CoreBuffers;
	/*! \brief Contains all functions and structure for computations variables */
	namespace compute
	{

		/*! \class Aberration
		**
		** Class that manages the correction of aberrations
		*/
		class Aberration
		{
		public:

			using ComplexArray = holovibes::cuda_tools::Array<cufftComplex>;
			using FloatArray = holovibes::cuda_tools::Array<float>;
			/** \brief Constructor.
			*/
			Aberration(const CoreBuffers& buffers,
				const camera::FrameDescriptor& fd,
				const holovibes::ComputeDescriptor& cd,
				ComplexArray& lens);

			/*! \brief Computes the corrections and apply them to the lens
			**
			** Should be called between the first lens computation and fft call
			*/
			void operator()();

		private:

			/** \brief Computes the shift of all the frames, and stores the outputs in shifts_
			*/
			void compute_all_shifts();

			/** \brief Computes the correlation between two buffers and write the result into convolution_.
			*/
			void compute_correlation(const float* x, const float *y);
			/** \brief Computes the convolution between two buffers and write the result into \param out
			*/
			void compute_convolution(const float* x, const float* y, float* out);

			/*! \brief Finds the position of the maximum in the correlation buffer
			*/
			QPoint find_maximum();


			/*! \brief Computes one phi to apply to the frame using the shifts
			*/
			cufftComplex compute_one_phi(QPoint point);


			/*! \brief Applies all the phis to the frames
			*/
			void apply_all_to_lens();

			//! Buffer to keep the reference frame (top left)
			FloatArray						ref_frame_;

			//! Buffer to keep the current frame compared to the reference
			FloatArray						frame_;

			//! Buffer to keep the correlation
			FloatArray						correlation_;

			//! 2D vector containing all the shifts detected
			std::vector<std::vector<QPoint>>	shifts_;

			//! Main buffers
			const CoreBuffers&				buffers_;

			//! Lens buffer
			ComplexArray&					lens_;

			//! Describes the frame size
			const camera::FrameDescriptor&	fd_;

			//! Compute Descriptor
			const ComputeDescriptor&		cd_;
		};
	}
}
