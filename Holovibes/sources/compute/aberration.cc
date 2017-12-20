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

}

void Aberration::compute_all_shifts()
{

}

void Aberration::compute_correlation(const float* x, const float *y)
{

}

void Aberration::compute_convolution(const float* x, const float* y, float* out)
{

}

QPoint Aberration::find_maximum()
{
	return {};
}

cufftComplex Aberration::compute_one_phi(QPoint point)
{
	return {};
}

void Aberration::apply_all_to_lens()
{

}


