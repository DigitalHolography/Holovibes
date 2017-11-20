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

#include "detect_intensity.hh"
#include "compute_descriptor.hh"
#include "rect.hh"
#include "power_of_two.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "stabilization.cuh"

#include <iostream>
#include <cufft.h>

using holovibes::compute::DetectIntensity;
using holovibes::FnVector;


DetectIntensity::DetectIntensity(FnVector& fn_vect,
	cuComplex* const& gpu_input_buffer,
	const camera::FrameDescriptor& fd,
	holovibes::ComputeDescriptor& cd)
	: last_intensity_(0)
	, current_shift_(0)
	, is_delaying_shift_(false)
	, fn_vect_(fn_vect)
	, gpu_input_buffer_(gpu_input_buffer)
	, fd_(fd)
	, cd_(cd)
{}

void DetectIntensity::insert_post_contiguous_complex()
{
	fn_vect_.push_back([=]() {
		check_jump();
		update_shift();
		update_lambda();
	});
}

void DetectIntensity::check_jump()
{
	float current_intensity = get_current_intensity();
	//std::cout << current_intensity << std::endl;
	if (is_jump(current_intensity, last_intensity_))
		on_jump();
	last_intensity_ = current_intensity;
}

bool DetectIntensity::is_jump(float current, float last)
{
	float threshold = cd_.interp_sensitivity;
	return current < threshold * last;
}

float DetectIntensity::get_current_intensity()
{
	float* buffer_ptr = reinterpret_cast<float*>(gpu_input_buffer_);
	const uint nb_pixels = fd_.frame_res() * 2;

	// Selecting 1/4 of the pixels positioned at the center
	buffer_ptr += nb_pixels * 3 / 8;
	auto res = average_operator(buffer_ptr, nb_pixels / 4);

	cudaStreamSynchronize(0);
	return res;
}

void DetectIntensity::update_shift()
{
	if (is_delaying_shift_)
	{
		if (current_shift_++ == cd_.interp_shift)
		{
			on_jump(true);
			is_delaying_shift_ = false;
		}
	}
}

void DetectIntensity::on_jump(bool delayed)
{
	if (cd_.interp_shift > 0 && !delayed)
	{
		if (!is_delaying_shift_)
		{
			is_delaying_shift_ = true;
			current_shift_ = 0;
		}
	}
	else
	{
		std::cout << "jump" << std::endl;
		frames_since_jump_ = 0;
	}
}

void DetectIntensity::update_lambda()
{
	frames_since_jump_++;
	float lambda = cd_.interp_lambda1.load();
	float progress = static_cast<float>(frames_since_jump_) / cd_.nsamples;
	if (progress > 1)
		progress = 1;
	lambda += (cd_.interp_lambda2 - cd_.interp_lambda1) * progress;
	cd_.interp_lambda = lambda;
	std::cout << lambda << std::endl;
}

