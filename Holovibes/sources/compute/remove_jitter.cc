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

# include "tools_compute.cuh"

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
		fn_vect_.push_back([this]() {
			perform_input_fft();
			compute_all_shifts();
			fix_jitter();
		});
	}
}

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

void RemoveJitter::correlation(cuComplex* ref, cuComplex* slice, float* out)
{
	auto size = slice_size();

	// The frames are already in frequency domain
	multiply_frames_complex(ref, slice, slice, size);
	cudaStreamSynchronize(0);

	CufftHandle plan2d(fd_.width, size, CUFFT_C2R);
	cufftExecC2R(plan2d, slice, out);
	cudaStreamSynchronize(0);
}

int RemoveJitter::maximum_y(float* frame)
{
	uint max = 0;
	gpu_extremums(frame, slice_size(), nullptr, nullptr, nullptr, &max);
	uint max_y = max / fd_.width;
	return max_y;
}

void RemoveJitter::fix_jitter()
{
	// Phi_jitter[i] = 2*PI/Lambda * Sum(n: 0 -> i, shift_t[n])
	int sum_phi = 0;
	std::vector<int> phi;
	for (size_t i = 0; i < shift_t_.size(); i++)
	{
		int phi_jitter = sum_phi + shift_t_[i];
		sum_phi = phi_jitter;
		phi_jitter *= M_PI_2 / cd_.lambda;
		phi.push_back(phi_jitter);
	}

	int small_chunk_size = cd_.nsamples / nb_slices_;
	int big_chunk_size = small_chunk_size * 1.5;

	// multiply all small chunks
	int chunk_no = 1;
	for (int p = big_chunk_size; p < cd_.nsamples - big_chunk_size; p += small_chunk_size, chunk_no++) {
		cuComplex multiplier;
		multiplier.x = cosf(-phi[chunk_no]);
		multiplier.y = sinf(-phi[chunk_no]);
		gpu_multiply_const(buffers_.gpu_input_buffer_.get() + fd_.frame_size() * p, fd_.frame_size() * small_chunk_size, multiplier);
	}

	// multiply the final big chunk
	cuComplex multiplier;
	multiplier.x = cosf(-phi.back());
	multiplier.y = sinf(-phi.back());
	gpu_multiply_const(buffers_.gpu_input_buffer_.get() + fd_.frame_size() * (cd_.nsamples - big_chunk_size), fd_.frame_size() * big_chunk_size, multiplier);
}

void RemoveJitter::compute_one_shift(int i)
{
	extract_and_fft(i, slice_);

	auto size = slice_size();
	correlation_.ensure_minimum_size(size);

	correlation(ref_slice_, slice_, correlation_);
	int max_y = maximum_y(correlation_);

	shift_t_.push_back(max_y);
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
