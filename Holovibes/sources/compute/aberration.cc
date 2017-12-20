#include "aberration.hh"
#include "compute_descriptor.hh"
#include "rect.hh"
#include "power_of_two.hh"
#include "cufft_handle.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "stabilization.cuh"
#include "icompute.hh"

using holovibes::compute::Aberration;
using holovibes::FnVector;
using holovibes::cuda_tools::CufftHandle;
using holovibes::CoreBuffers;
using ComplexArray = Aberration::ComplexArray;


Aberration::Aberration(const CoreBuffers& buffers,
	const camera::FrameDescriptor& fd,
	const holovibes::ComputeDescriptor& cd,
	ComplexArray& lens)
	: buffers_(buffers)
	, lens_(lens)
	, fd_(fd)
	, cd_(cd)
{}

void Aberration::operator()()
{
	compute_all_shifts();
	apply_all_to_lens();
}

void Aberration::extract_and_fft(uint x_index, uint y_index, float* buffer)
{

}

QPoint Aberration::compute_one_shift(uint x, uint y)
{
	extract_and_fft(x, y, frame_);
	compute_correlation(ref_frame_, frame_);
	return find_maximum();
}

void Aberration::compute_all_shifts()
{
	extract_and_fft(0, 0, ref_frame_.get());
	rotation_180(ref_frame_.get(), { frame_size_.x, frame_size_.y });

	shifts_.clear();
	for (uint i = 0; i < nb_frames_; ++i)
	{
		std::vector<QPoint> column;
		for (uint j = 0; j < nb_frames_; ++j)
			column.push_back(compute_one_shift(i, j));
		shifts_.push_back(column);
	}

}

void Aberration::compute_correlation(const float* x, const float *y)
{

}

QPoint Aberration::find_maximum()
{
	uint max = 0;
	uint frame_res = frame_size_.x * frame_size_.y;
	gpu_extremums(correlation_.get(), frame_res, nullptr, nullptr, nullptr, &max);
	// x y: Coordinates of maximum of the correlation function
	int x = max % frame_size_.x;
	int y = max / frame_size_.x;
	if (x > frame_size_.x / 2)
		x -= frame_size_.x;
	if (y > frame_size_.y / 2)
		y -= frame_size_.y;

	return {x, y};
}

cufftComplex Aberration::compute_one_phi(QPoint point)
{
	return {};
}

void Aberration::apply_all_to_lens()
{

}


