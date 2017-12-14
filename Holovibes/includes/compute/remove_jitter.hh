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

			void extract_and_fft(int slice_index, cuComplex* buffer);
			void correlation(cuComplex* ref, cuComplex* slice, float* out);
			int maximum_y(float* frame);
			int compute_one_shift(int i);
			void compute_all_shifts();
			void fix_jitter();
			int slice_size();



			uint							nb_slices_{ 7 };
			uint							slice_depth_;
			uint							slice_shift_;



			cuda_tools::Array<cuComplex>	ref_slice_;
			cuda_tools::Array<cuComplex>	slice_;
			cuda_tools::Array<float>		correlation_;

			std::vector<int>				shift_t_;

			cuComplex*						buffer_;
			const units::RectFd&			dimensions_;

			const ComputeDescriptor&		cd_;
		};
	}
}
