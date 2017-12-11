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
	struct CoreBuffers;

	namespace compute
	{

		class RemoveJitter
		{
		public:
			RemoveJitter(FnVector& fn_vect,
				const CoreBuffers& buffers,
				const camera::FrameDescriptor& fd,
				const holovibes::ComputeDescriptor& cd);

			void insert_pre_fft();

		private:

			void extract_and_fft(int slice_index, cuComplex* buffer);
			void extract_input_frame();
			void perform_input_fft();
			void correlation(cuComplex* ref, cuComplex* slice, float* out);
			int maximum_y(float* frame);
			void compute_one_shift(int i);
			void compute_all_shifts();
			void fix_jitter();
			void fft(cuComplex* from, cuComplex* to, int direction);
			int slice_size();



			float							slice_overlap_coeff_ {1.f};
			int								slice_shift_ {4};
			int								nb_slices_;



			cuda_tools::Array<cuComplex>	fft_frame_;
			cuda_tools::Array<cuComplex>	ref_slice_;
			cuda_tools::Array<cuComplex>	slice_;
			cuda_tools::Array<float>		correlation_;

			std::vector<int>				shift_t_;

			FnVector&						fn_vect_;
			const CoreBuffers&				buffers_;
			const camera::FrameDescriptor&	fd_;

			const ComputeDescriptor&		cd_;
		};
	}
}
