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
{}

void RemoveJitter::insert_pre_fft()
{
	fn_vect_.push_back([=]() {
		perform_fft();
		compute_all_shifts();
		fix_jitter();
	});
}


void RemoveJitter::extract_slice(int slice_index, cuComplex* buffer)
{

}

void RemoveJitter::perform_fft()
{

}

void RemoveJitter::correlation()
{

}

void RemoveJitter::fix_jitter()
{

}

void RemoveJitter::compute_all_shifts()
{

}
