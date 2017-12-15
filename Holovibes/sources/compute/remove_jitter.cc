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
#include "icompute.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "Common.cuh"
#include "average.cuh"
#include "stabilization.cuh"

using holovibes::compute::RemoveJitter;
using holovibes::FnVector;
using holovibes::cuda_tools::CufftHandle;
using holovibes::cuda_tools::Array;
using holovibes::Stft_env;
using holovibes::units::RectFd;

RemoveJitter::RemoveJitter(cuComplex* buffer,
	const RectFd& dimensions,
	const holovibes::ComputeDescriptor& cd)
	: buffer_(buffer)
	, dimensions_(dimensions)
	, cd_(cd)
{
	nb_slices_ = std::max(3, cd.jitter_slices_.load());
	if (!nb_slices_ % 2)
		nb_slices_++;
	slice_depth_ = cd_.nsamples /((nb_slices_ + 1) / 2);
	slice_shift_ = slice_depth_ / 2;

	auto size = slice_size();
	ref_slice_.resize(size);
	slice_.resize(size);
	correlation_.resize(size);
}


int RemoveJitter::slice_size()
{
	return dimensions_.width() * slice_depth_;
}


void RemoveJitter::extract_and_fft(int slice_index, cuComplex* buffer)
{
	const int width = dimensions_.width();
	int depth = slice_depth_;
	const int frame_size = dimensions_.area();
	CufftHandle plan1d;
	plan1d.planMany(1, &depth,
		&depth, frame_size, 1,
		&depth, width, 1,
		CUFFT_C2C, width);


	// Trasform t -> w of the slice
	auto in = buffer_ + cd_.getStftCursor().y() * width;
	const int pixel_shift_depth = slice_index * frame_size * slice_shift_;
	in += pixel_shift_depth;
	cufftExecC2C(plan1d, in, buffer, CUFFT_FORWARD);
	cudaStreamSynchronize(0);
	cudaCheckError();

	// Preparing for the correlation

	normalize_frame(reinterpret_cast<float*>(buffer), 2 * slice_size());

	plan1d.planMany(1, &depth,
		&depth, width, 1,
		&depth, width, 1,
		CUFFT_C2C, width);
	cufftExecC2C(plan1d, buffer, buffer, CUFFT_FORWARD);
	cudaStreamSynchronize(0);
	cudaCheckError();
}

void RemoveJitter::correlation(cuComplex* ref, cuComplex* slice, float* out)
{
	auto size = slice_size();

	// The frames are already in frequency domain
	multiply_frames_complex(ref, slice, slice, size);
	cudaStreamSynchronize(0);

	int width = dimensions_.width();
	int depth = slice_depth_;
	CufftHandle plan1d;

	plan1d.planMany(1, &depth,
		&depth, width, 1,
		&depth, width, 1,
		CUFFT_C2R, width);

	cufftExecC2R(plan1d, slice, out);
	cudaStreamSynchronize(0);
	cudaCheckError();
}

int RemoveJitter::maximum_y(float* frame)
{
	Array<float> line_averages(slice_depth_);
	average_lines(frame, line_averages, dimensions_.width(), slice_depth_);
	cudaStreamSynchronize(0);

	//float tmp[1024];
	//cudaMemcpy(tmp, line_averages.get(), slice_depth_ * sizeof(float), cudaMemcpyDeviceToHost);

	uint max_y = 0;
	gpu_extremums(line_averages, slice_depth_, nullptr, nullptr, nullptr, &max_y);
	if (max_y > slice_depth_ / 2)
		max_y -= slice_depth_;
	return max_y;
}

int RemoveJitter::compute_one_shift(int i)
{
	extract_and_fft(i, slice_);

	correlation(ref_slice_, slice_, correlation_);
	int max_y = maximum_y(correlation_);

	//correlation_.write_to_file("H:/test.raw");
	//std::cout << dimensions_.width() << std::endl;

	cudaCheckError();
	return max_y;
}

void RemoveJitter::compute_all_shifts()
{
	extract_and_fft(0, ref_slice_);

	// We flip it for the convolution, to do it only once
	rotation_180(ref_slice_.get(), {dimensions_.width(), static_cast<int>(slice_depth_)});

	shifts_.clear();
	for (uint i = 0; i < nb_slices_; ++i)
		shifts_.push_back(compute_one_shift(i));

	for (auto i : shifts_)
		std::cout << std::setw(3) << i << " ";
	std::cout << std::endl;
}

void RemoveJitter::fix_jitter()
{
	// Phi_jitter[i] = 2*PI/Lambda * Sum(n: 0 -> i, shift_t[n])
	std::vector<double> phi;
	for (size_t i = 1; i < shifts_.size(); i++)
	{
		double current_phi = shifts_[i];
		current_phi *= M_PI * 2 / cd_.lambda;
		//std::cout << phi_jitter << " ";
		phi.push_back(current_phi);
	}
	//std::cout << std::endl;

	int big_chunk_size = slice_shift_ * 1.5;

	// multiply all small chunks
	int sign = cd_.jitter_enabled_ ? 1 : -1;
	int p = big_chunk_size;
	for (uint i = 0; i < phi.size() - 1; p += slice_shift_, i++) {
		cuComplex multiplier;
		multiplier.x = cosf(sign * phi[i]);
		multiplier.y = sinf(sign * phi[i]);
		gpu_multiply_const(buffer_ + dimensions_.area() * p, dimensions_.area() * slice_shift_, multiplier);
	}

	// multiply the final big chunk
	cuComplex multiplier;
	multiplier.x = cosf(sign * phi.back());
	multiplier.y = sinf(sign * phi.back());
	gpu_multiply_const(buffer_ + dimensions_.area() * p, dimensions_.area() * (cd_.nsamples - p), multiplier);
}

void RemoveJitter::run()
{
	if (cd_.stft_view_enabled && (true || cd_.jitter_enabled_))
	{
		compute_all_shifts();
		fix_jitter();
	}
}
