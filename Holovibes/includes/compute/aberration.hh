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

			//! refreshes with new values from compute descriptor, should be called on pipe reset
			void refresh();

		private:

			/*! Returns the number of pixels in one chunk
			 */
			uint chunk_area();
			int chunk_width();
			int chunk_height();


			/*! Extract one chunk from the buffer while performing the 2d fft
			 */
			void extract_and_fft(uint x_index, uint y_index, float* buffer);

			/** \brief Computes the shift of the chunk (x, y)
			*/
			QPoint compute_one_shift(uint x, uint y);

			/** \brief Computes the shift of all the chunks, and stores the outputs in shifts_
			*/
			void compute_all_shifts();

			/** \brief Computes the correlation between two buffers and write the result into convolution_.
			*/
			void compute_correlation(float* x, float *y);

			/*! \brief Finds the position of the maximum in the correlation buffer
			*/
			QPoint find_maximum();


			/*! \brief Computes one phi to apply to the chunk using the shifts
			*/
			cufftComplex compute_one_phi(QPoint point);


			/*! \brief Applies all the phis to the chunks
			*/
			void apply_all_to_lens();

			//! Buffer to keep the reference chunk (top left)
			FloatArray						ref_chunk_;

			//! Buffer to keep the current chunk compared to the reference
			FloatArray						chunk_;

			//! Buffer to keep the correlation
			FloatArray						correlation_;

			//! 2D vector containing all the shifts detected
			std::vector<std::vector<QPoint>>	shifts_;

			//! Main buffers
			const CoreBuffers&				buffers_;

			//! Lens buffer
			ComplexArray&					lens_;

			//! Describes the chunk size
			const camera::FrameDescriptor&	fd_;

			//! Compute Descriptor
			const ComputeDescriptor&		cd_;

			uint							nb_chunks_;

			QPoint							chunk_size_;
		};
	}
}
