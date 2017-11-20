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
	, sum_frames_(0)
	, nb_jumps_(0)
	, fn_vect_(fn_vect)
	, gpu_input_buffer_(gpu_input_buffer)
	, fd_(fd)
	, cd_(cd)
{}

void DetectIntensity::insert_post_contiguous_complex()
{
	fn_vect_.push_back([=]() {
		check_jump();
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
	auto res = average_operator(reinterpret_cast<const float*>(gpu_input_buffer_), fd_.frame_res() * 2);
	cudaStreamSynchronize(0);
	return res;
}

void DetectIntensity::on_jump()
{
	std::cout << "jump" << std::endl;
	sum_frames_ += frames_since_jump_;
	frames_since_jump_ = 0;
	nb_jumps_++;
}

void DetectIntensity::update_lambda()
{
	frames_since_jump_++;
	float lambda = cd_.interp_lambda1.load();
	if (nb_jumps_)
	{
		const float average_frames = sum_frames_ / nb_jumps_;
		float progress = static_cast<float>(frames_since_jump_) / average_frames;
		if (progress > 1)
			progress = 1;
		lambda += (cd_.interp_lambda2 - cd_.interp_lambda1) * progress;
	}
	cd_.interp_lambda = lambda;
}

