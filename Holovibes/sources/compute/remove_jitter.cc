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

#include "remove_jitter.hh"
#include "compute_descriptor.hh"
#include "rect.hh"
#include "cufft_handle.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "icompute.hh"


using holovibes::compute::RemoveJitter;
using holovibes::FnVector;
using holovibes::cuda_tools::CufftHandle;
using holovibes::CoreBuffers;

RemoveJitter::RemoveJitter(FnVector& fn_vect,
	const CoreBuffers& buffers,
	const camera::FrameDescriptor& fd,
	const holovibes::ComputeDescriptor& cd)
	: fn_vect_(fn_vect)
	, buffers_(buffers)
	, fd_(fd)
	, cd_(cd)
{
}

void RemoveJitter::insert_pre_fft()
{
	if (cd_.stft_view_enabled)
	{
		fn_vect_.push_back([=]() {
			perform_input_fft();
			compute_all_shifts();
			fix_jitter();
		});
	}
}

// Could be removed, there should be a way to fft directly
void RemoveJitter::extract_input_frame()
{
	fft_frame_.ensure_minimum_size(fd_.frame_res());
	auto line = buffers_.gpu_input_buffer_.get();
	line += cd_.getStftCursor().x() * fd_.width;
	auto frame_size = fd_.frame_res();
	auto buffer_ptr = fft_frame_.get();
	for (int i = 0; i < cd_.nsamples; ++i)
	{
		cudaMemcpyAsync(buffer_ptr, line, fd_.width * sizeof(cuComplex), cudaMemcpyDeviceToDevice, 0);
		buffer_ptr += fd_.width;
		line += frame_size;
	}
	cudaStreamSynchronize(0);
}

void RemoveJitter::perform_input_fft()
{
	extract_input_frame();
	CufftHandle plan2d(fd_.width, fd_.height, CUFFT_C2C);
	cufftExecC2C(plan2d, fft_frame_, fft_frame_, CUFFT_FORWARD);
	cudaStreamSynchronize(0);
}

void RemoveJitter::extract_and_fft(int slice_index, cuComplex* buffer)
{
	auto src = fft_frame_.get() + slice_size() * slice_index;
	// TODO check when going out of bounds (needs to loop)

	fft(src, buffer, CUFFT_FORWARD);
}

int RemoveJitter::correlation(cuComplex* ref, cuComplex* slice)
{
	return 0;
	// FIXME
}

void RemoveJitter::fix_jitter()
{

}

void RemoveJitter::compute_one_shift(int i)
{
	extract_and_fft(i, slice_);
	int c = correlation(ref_slice_, slice_);
	shift_t_.push_back(c);
}

void RemoveJitter::compute_all_shifts()
{
	auto size = slice_size();
	ref_slice_.ensure_minimum_size(size);
	slice_.ensure_minimum_size(size);

	extract_and_fft(0, ref_slice_);

	shift_t_.clear();
	for (int i = 0; i < nb_slices_; ++i)
		compute_one_shift(i);
}

void RemoveJitter::fft(cuComplex* from, cuComplex* to, int direction)
{
	CufftHandle plan2d(fd_.width, slice_size(), CUFFT_C2C);
	cufftExecC2C(plan2d, from, to, direction);
	cudaStreamSynchronize(0);
}

int RemoveJitter::slice_size()
{
	return fd_.width / slice_shift_;
}
