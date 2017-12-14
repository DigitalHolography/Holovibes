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
 * Contains functions relative jitter cancel. */
#pragma once

# include "cuda_tools/unique_ptr.hh"
# include "cuda_tools/array.hh"
# include "pipeline_utils.hh"
# include "frame_desc.hh"
# include "rect.hh"

namespace holovibes
{
	class ComputeDescriptor;
	struct Stft_env;

	namespace compute
	{

		class RemoveJitter
		{
		public:
			RemoveJitter(cuComplex* buffer,
				const units::RectFd& dimensions,
				const holovibes::ComputeDescriptor& cd);

			void run();

		private:

			/*! Returns the size in pixel of one slice (width * depth)
			*/
			int slice_size();

			/*! Extract one frame from the buffrt while performing the fft,
			 * and fft again for the correlation later on
			 */
			void extract_and_fft(int slice_index, cuComplex* buffer);

			/*! Compute the correlation between the two buffers using ffts
			* the first fft is assumed to be already done
			*/
			void correlation(cuComplex* ref, cuComplex* slice, float* out);

			/*! Average each line and finds the index of the maximum*/
			int maximum_y(float* frame);

			/*! Compute one shift, extracting the frame, correlating it with the reference */
			int compute_one_shift(int i);

			/*! Fill the shifts vector with the shift of each frame
			*
			* Doesn't include the first frame, it's always 0
			*/
			void compute_all_shifts();

			/*! Use the shifts vector to correct the input buffer
			*/
			void fix_jitter();



			/*! Number of slices we split the input into, must be odd
			*/
			uint							nb_slices_{ 7 };

			/*! Size of a slice in the time axis
			*/
			uint							slice_depth_;

			/*! Difference between the start of two consecutive slices (depth / 2)
			*/
			uint							slice_shift_;



			/*! Buffer containing the reference slice (first one)
			*/
			cuda_tools::Array<cuComplex>	ref_slice_;

			/*! Buffer containing the current slice
			*/
			cuda_tools::Array<cuComplex>	slice_;

			/*! Buffer containing correlation result
			*/
			cuda_tools::Array<float>		correlation_;

			/*! Vector containing all the shifts
			*/
			std::vector<int>				shifts_;

			/*! Pointer to the 3d buffer, assumed to be the cropped stft buffer
			*/
			cuComplex*						buffer_;

			/*! Cropped stft area, we only use width() and height() here, not its position
			*/
			const holovibes::units::RectFd&	dimensions_;

			/*! Compute descriptor
			*/
			const ComputeDescriptor&		cd_;
		};
	}
}
