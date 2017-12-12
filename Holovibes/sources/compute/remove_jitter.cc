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

#include "tools_compute.cuh"
#include "Common.cuh"
#include "average.cuh"

using holovibes::compute::RemoveJitter;
using holovibes::FnVector;
using holovibes::cuda_tools::CufftHandle;
using holovibes::cuda_tools::Array;
using holovibes::Stft_env;

RemoveJitter::RemoveJitter(FnVector& fn_vect,
	const Stft_env& buffers,
	const camera::FrameDescriptor& fd,
	const holovibes::ComputeDescriptor& cd)
	: fn_vect_(fn_vect)
	, buffers_(buffers)
	, fd_(fd)
	, cd_(cd)
{
	slice_depth_ = cd_.nsamples /((nb_slices_ + 1) / 2);
	slice_shift_ = slice_depth_ / 2;
}

void RemoveJitter::insert_pre_fft()
{
	if (cd_.stft_view_enabled && (true || cd_.jitter_enabled_))
	{
		fn_vect_.push_back([this]() {
			if (buffers_.stft_frame_counter_ == cd_.stft_steps)
			{
				compute_all_shifts();
				fix_jitter();
			}
		});
	}
}


void RemoveJitter::extract_and_fft(int slice_index, cuComplex* buffer)
{
	int width = fd_.width;
	int depth = slice_depth_;
	int frame_size = fd_.frame_res();
	CufftHandle plan1d;
	plan1d.planMany(1, &depth,
		&depth, frame_size, 1,
		&depth, width, 1,
		CUFFT_C2C, width);


	auto in = buffers_.gpu_stft_buffer_.get() + cd_.getStftCursor().y() * fd_.width;
	int pixel_shift_depth = slice_index * frame_size * slice_shift_;
	in += pixel_shift_depth;
	cufftExecC2C(plan1d, in, buffer, CUFFT_FORWARD);
	cudaStreamSynchronize(0);
	cudaCheckError();

	// Preparing for the convolution
	CufftHandle plan2d(width, depth, CUFFT_C2C);
	cufftExecC2C(plan2d, buffer, buffer, CUFFT_FORWARD);
	cudaStreamSynchronize(0);
	cudaCheckError();
}

void RemoveJitter::correlation(cuComplex* ref, cuComplex* slice, float* out)
{
	auto size = slice_size();

	// The frames are already in frequency domain
	multiply_frames_complex(ref, slice, slice, size);
	cudaStreamSynchronize(0);

	int width = fd_.width;
	int depth = slice_depth_;
	CufftHandle plan1d;

	plan1d.plan(width, depth, CUFFT_C2R);

	cufftExecC2R(plan1d, slice, out);
	cudaStreamSynchronize(0);
	cudaCheckError();
}

int RemoveJitter::maximum_y(float* frame)
{
	Array<float> line_averages(slice_depth_);
	average_lines(frame, line_averages, fd_.width, slice_depth_);
	cudaStreamSynchronize(0);

	float test[16];
	cudaMemcpy(test, line_averages, 64, cudaMemcpyDeviceToHost);

	uint max_y = 0;
	gpu_extremums(line_averages, slice_depth_, nullptr, nullptr, nullptr, &max_y);
	if (max_y > slice_depth_ / 2)
		max_y -= slice_depth_;
	return max_y;
}

void RemoveJitter::fix_jitter()
{
	// Phi_jitter[i] = 2*PI/Lambda * Sum(n: 0 -> i, shift_t[n])
	int sum_phi = 0;
	std::vector<double> phi;
	for (size_t i = 0; i < shift_t_.size(); i++)
	{
		double phi_jitter = sum_phi + shift_t_[i];
		sum_phi = phi_jitter;
		phi_jitter *= M_PI_2 / cd_.lambda;
		//std::cout << phi_jitter << " ";
		phi.push_back(phi_jitter);
	}
	//std::cout << std::endl;

	int big_chunk_size = slice_shift_ * 1.5;

	// multiply all small chunks
	int chunk_no = 1;
	int sign = cd_.jitter_enabled_ ? 1 : -1;
	for (int p = big_chunk_size; p < cd_.nsamples - big_chunk_size; p += slice_shift_, chunk_no++) {
		cuComplex multiplier;
		multiplier.x = cosf(sign * phi[chunk_no]);
		multiplier.y = sinf(sign * phi[chunk_no]);
		gpu_multiply_const(buffers_.gpu_stft_buffer_.get() + fd_.frame_size() * p, fd_.frame_size() * slice_shift_, multiplier);
	}

	// multiply the final big chunk
	cuComplex multiplier;
	multiplier.x = cosf(sign * phi.back());
	multiplier.y = sinf(sign * phi.back());
	gpu_multiply_const(buffers_.gpu_stft_buffer_.get() + fd_.frame_size() * (cd_.nsamples - big_chunk_size), fd_.frame_size() * big_chunk_size, multiplier);
}

int RemoveJitter::compute_one_shift(int i)
{
	extract_and_fft(i, slice_);

	auto size = slice_size();
	correlation_.ensure_minimum_size(size);

	correlation(ref_slice_, slice_, correlation_);
	int max_y = maximum_y(correlation_);

	cudaCheckError();
	return max_y;
}

void RemoveJitter::compute_all_shifts()
{
	auto size = slice_size();
	ref_slice_.ensure_minimum_size(size);
	slice_.ensure_minimum_size(size);

	extract_and_fft(0, ref_slice_);

	shift_t_.clear();
	for (int i = 1; i < nb_slices_; ++i)
		shift_t_.push_back(compute_one_shift(i));

	for (auto i : shift_t_)
		std::cout << std::setw(2) << i << " ";
	std::cout << std::endl;
}


int RemoveJitter::slice_size()
{
	return fd_.width * slice_depth_;
}
