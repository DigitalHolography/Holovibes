#include "aberration.hh"
#include "compute_descriptor.hh"
#include "rect.hh"
#include "power_of_two.hh"
#include "cufft_handle.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "aberration.cuh"
#include "icompute.hh"
#include "stabilization.cuh"

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
{
	refresh();
}

void Aberration::refresh()
{
	if (cd_.aberration_enabled_)
	{
		nb_frames_ = cd_.aberration_slices_;
		frame_size_.setX(fd_.width / nb_frames_);
		frame_size_.setY(fd_.height / nb_frames_);
		const auto area = frame_area();
		if (!ref_frame_.ensure_minimum_size(area)
			|| !frame_.ensure_minimum_size(area)
			|| !correlation_.ensure_minimum_size(area))
			throw std::bad_alloc();
	}
	else
	{
		ref_frame_.reset();
		frame_.reset();
		correlation_.reset();
	}
}

void Aberration::operator()()
{
	if (cd_.aberration_enabled_)
	{
		if (!frame_)
			refresh();
		compute_all_shifts();
		apply_all_to_lens();
	}
}

uint Aberration::frame_area()
{
	return frame_width() * frame_height();
}
int Aberration::frame_width()
{
	return frame_size_.x();
}

int Aberration::frame_height()
{
	return frame_size_.y();
}

void Aberration::extract_and_fft(uint x_index, uint y_index, float* buffer)
{
	auto nb_chunks_per_row = nb_frames_;
	auto nb_chunks_per_column = nb_frames_;

	auto full_chunk_width = fd_.width / nb_chunks_per_row;
	auto full_chunk_height = fd_.height / nb_chunks_per_column;
	auto nb_chunks = nb_chunks_per_row * nb_chunks_per_column;
	auto cropped_chunk_width = full_chunk_width * chunk_border_;
	auto cropped_chunk_height = full_chunk_height * chunk_border_;
	auto cropped_chunk_size = cropped_chunk_width * cropped_chunk_height;
	int n[2] = { cropped_chunk_height, cropped_chunk_width };
	
	// Indicates the distance between the first element of two consecutive signals in a batch of the input data
	//int idist = cropped_chunk_width + ;
	int inembed[2] = {1, fd_.width};

	CufftHandle plan;
	cufftPlanMany(&plan.get(), 2, n, inembed, 1, full_chunk_width, nullptr, 1, cropped_chunk_size, CUFFT_C2R, nb_chunks);

    cufftExecC2R(plan, , , CUFFT_FORWARD);
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
	rotation_180(ref_frame_.get(), { frame_width(), frame_height() });

	shifts_.clear();
	for (uint i = 0; i < nb_frames_; ++i)
	{
		std::vector<QPoint> column;
		for (uint j = 0; j < nb_frames_; ++j)
			column.push_back(compute_one_shift(i, j));
		shifts_.push_back(column);
	}

}

void Aberration::compute_correlation(float* x, float *y)
{
	correlation_operator(x, y, correlation_, { frame_width(), frame_height() });
}

QPoint Aberration::find_maximum()
{
	uint max = 0;
	uint frame_res = frame_area();
	gpu_extremums(correlation_.get(), frame_res, nullptr, nullptr, nullptr, &max);
	// x y: Coordinates of maximum of the correlation function
	int x = max % frame_width();
	int y = max / frame_width();
	if (x > frame_width() / 2)
		x -= frame_width();
	if (y > frame_height() / 2)
		y -= frame_height();

	return {x, y};
}

cufftComplex Aberration::compute_one_phi(QPoint point)
{
	return {};
}

void Aberration::apply_all_to_lens()
{
	std::vector<cufftComplex> phis;
	for (uint i = 0; i < nb_frames_; i++) {
		for (uint j = 0; j < nb_frames_; j++) {
			phis.push_back(compute_one_phi(shifts_[i][j]));
		}
	}
	apply_aberration_phis(lens_, phis, nb_frames_, nb_frames_, fd_);
}


