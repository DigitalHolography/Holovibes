#include "aberration.hh"
#include "compute_descriptor.hh"
#include "rect.hh"
#include "power_of_two.hh"
#include "cufft_handle.hh"
#include "array.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "aberration.cuh"
#include "icompute.hh"
#include "stabilization.cuh"

using holovibes::compute::Aberration;
using holovibes::FnVector;
using holovibes::cuda_tools::CufftHandle;
using holovibes::cuda_tools::Array;
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
		nb_chunks_ = cd_.aberration_slices_;
		chunk_size_.setX(fd_.width / nb_chunks_);
		chunk_size_.setY(fd_.height / nb_chunks_);
		const auto area = chunk_area();
		if (!ref_chunk_.ensure_minimum_size(area)
			|| !chunk_.ensure_minimum_size(area)
			|| !correlation_.ensure_minimum_size(area))
			throw std::bad_alloc();
	}
	else
	{
		ref_chunk_.reset();
		chunk_.reset();
		correlation_.reset();
	}
}

void Aberration::operator()()
{
	if (cd_.aberration_enabled_)
	{
		if (!chunk_)
			refresh();
		compute_all_shifts();
		apply_all_to_lens();
	}
}

uint Aberration::chunk_area()
{
	return chunk_width() * chunk_height();
}
int Aberration::chunk_width()
{
	return chunk_size_.x();
}

int Aberration::chunk_height()
{
	return chunk_size_.y();
}

void Aberration::extract_and_fft(uint x_index, uint y_index, float* buffer)
{
	cuComplex* input = buffers_.gpu_input_buffer_;
	for (uint i = 0; i < chunk_height(); i++)
	{
		cudaMemcpyAsync(buffer + i * chunk_width(), input, chunk_width() * sizeof(cuComplex), cudaMemcpyDeviceToDevice, 0);
		input += fd_.width;
	}
	cudaStreamSynchronize(0);
	cudaCheckError();

	Array<cuComplex> tmp_complex_buffer(chunk_area());
	CufftHandle plan2d(chunk_width(), chunk_height(), CUFFT_C2C);
	cufftExecC2C(plan2d, tmp_complex_buffer, tmp_complex_buffer, CUFFT_FORWARD);
	cudaStreamSynchronize(0);
	cudaCheckError();

	complex_to_modulus(tmp_complex_buffer, buffer, nullptr, 0, 0, chunk_area());
	cudaStreamSynchronize(0);
	cudaCheckError();

	normalize_frame(buffer, chunk_area());
}

QPoint Aberration::compute_one_shift(uint x, uint y)
{
	extract_and_fft(x, y, chunk_);
	compute_correlation(ref_chunk_, chunk_);
	return find_maximum();
}

void Aberration::compute_all_shifts()
{
	extract_and_fft(0, 0, ref_chunk_.get());
	rotation_180(ref_chunk_.get(), { chunk_width(), chunk_height() });

	shifts_.clear();
	for (uint i = 0; i < nb_chunks_; ++i)
	{
		std::vector<QPoint> column;
		for (uint j = 0; j < nb_chunks_; ++j)
			column.push_back(compute_one_shift(i, j));
		shifts_.push_back(column);
	}

}

void Aberration::compute_correlation(float* x, float *y)
{
	correlation_operator(x, y, correlation_, { chunk_width(), chunk_height() });
}

QPoint Aberration::find_maximum()
{
	uint max = 0;
	uint chunk_res = chunk_area();
	gpu_extremums(correlation_.get(), chunk_res, nullptr, nullptr, nullptr, &max);
	// x y: Coordinates of maximum of the correlation function
	int x = max % chunk_width();
	int y = max / chunk_width();
	if (x > chunk_width() / 2)
		x -= chunk_width();
	if (y > chunk_height() / 2)
		y -= chunk_height();

	return {x, y};
}

cufftComplex Aberration::compute_one_phi(QPoint point)
{
	return {};
}

void Aberration::apply_all_to_lens()
{
	std::vector<cufftComplex> phis;
	for (uint i = 0; i < nb_chunks_; i++) {
		for (uint j = 0; j < nb_chunks_; j++) {
			phis.push_back(compute_one_phi(shifts_[i][j]));
		}
	}
	apply_aberration_phis(lens_, phis, nb_chunks_, nb_chunks_, fd_);
}


