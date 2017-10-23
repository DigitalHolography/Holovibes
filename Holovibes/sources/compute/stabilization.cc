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

#include "stabilization.hh"
#include "compute_descriptor.hh"
#include "Rectangle.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "stabilization.cuh"

#include <iostream>
#include <cufft.h>

using holovibes::compute::Stabilization;


Stabilization::Stabilization(FnVector& fn_vect,
	float* const& gpu_float_buffer,
	const camera::FrameDescriptor& fd,
	const holovibes::ComputeDescriptor& cd)
	: fn_vect_(fn_vect)
	, gpu_float_buffer_(gpu_float_buffer)
	, fd_(fd)
	, cd_(cd)
{}

void Stabilization::insert_post_img_type()
{
	if (cd_.xy_stabilization_enabled.load())
	{
		insert_convolution();
		insert_extremums();
		insert_stabilization();
	}
	insert_average();
}

void Stabilization::insert_convolution()
{
	fn_vect_.push_back([=]() {
		gui::Rectangle zone = cd_.getStabilizationZone();
		auto frame_res = fd_.frame_res();
		if (last_frame_.is_large_enough(frame_res))
		{
			convolution_.ensure_minimum_size(zone.area());
			compute_convolution(gpu_float_buffer_, last_frame_.get(), convolution_.get());
		}
		else
			last_frame_.resize(frame_res);
		cudaMemcpy(last_frame_.get(), gpu_float_buffer_, frame_res * sizeof(float), cudaMemcpyDeviceToDevice);
	});
}

void Stabilization::compute_convolution(const float* x, const float* y, float* out)
{
	cufftHandle plan2d_a;
	cufftHandle plan2d_b;
	cufftHandle plan2d_inverse;

	gui::Rectangle zone = cd_.getStabilizationZone();
	CudaUniquePtr<float> selected_x(zone.area());
	CudaUniquePtr<float> selected_y(zone.area());
	if (!selected_x || !selected_y)
		return;

	extract_frame(x, selected_x.get(), fd_.width, zone);
	extract_frame(y, selected_y.get(), fd_.width, zone);


	cufftPlan2d(&plan2d_a, zone.height(), zone.height(), CUFFT_R2C);
	cufftPlan2d(&plan2d_b, zone.height(), zone.width(), CUFFT_R2C);
	cufftPlan2d(&plan2d_inverse, zone.height(), zone.width(), CUFFT_C2R);

	gpu_float_divide(selected_x.get(), zone.area(), 65536);
	gpu_float_divide(selected_y.get(), zone.area(), 65536);
	constexpr uint s = 64;
	convolution_float(
		selected_x.get(),
		selected_y.get(),
		out,
		zone.area(),
		plan2d_a,
		plan2d_b,
		plan2d_inverse);
	float _x[s];
	cudaMemcpy(_x, selected_x.get(), s * 4, cudaMemcpyDeviceToHost);
	float _y[s];
	cudaMemcpy(_y, selected_y.get(), s * 4, cudaMemcpyDeviceToHost);
	float _conv[s];
	cudaMemcpy(_conv, convolution_.get(), s * 4, cudaMemcpyDeviceToHost);
	cufftDestroy(plan2d_a);
	cufftDestroy(plan2d_b);
	cufftDestroy(plan2d_inverse);
}

void Stabilization::insert_extremums()
{
	fn_vect_.push_back([=]() {
		gui::Rectangle zone = cd_.getStabilizationZone();
		const auto frame_res = zone.area();
		if (convolution_.is_large_enough(frame_res))
		{
			uint max = 0;
			gpu_extremums(convolution_.get(), frame_res, nullptr, nullptr, nullptr, &max);
			// x y: Coordinates of maximum of the correlation function
			int x = max % zone.width();
			int y = max / zone.width();
			if (x > zone.width() / 2)
				x -= zone.width();
			if (y > zone.height() / 2)
				y -= zone.height();
			//shift_x = (shift_x + x + fd.width) % fd.width;
			//shift_y = (shift_y + y + fd.height) % fd.height;

			static int old_x = 0;
			static int old_y = 0;

			shift_x = old_x - x;
			shift_y = old_y - y;

			old_x = x;
			old_y = y;

			std::cout << shift_x << ", " << shift_y << std::endl;
		}
	});
}


void Stabilization::insert_stabilization()
{
	if (cd_.xy_stabilization_show_convolution.load())
	{
		// Visualization of convolution matrix
		fn_vect_.push_back([=]()
		{
			gui::Rectangle zone = cd_.getStabilizationZone();
			if (convolution_.is_large_enough(zone.area()))
			{
				gpu_resize(convolution_.get(), gpu_float_buffer_, { zone.width(), zone.height() }, { fd_.width, fd_.height });
			}
		});
	}
	else
	{
		// Visualization of image
		fn_vect_.push_back([=]()
		{
			complex_translation(gpu_float_buffer_, fd_.width, fd_.height, shift_x, shift_y);
		});
	}
}



void Stabilization::insert_average()
{
	bool queue_needed = cd_.img_acc_slice_xy_enabled || cd_.xy_stabilization_enabled;
	if (queue_needed)
	{
		if (!accumulation_queue_ || cd_.img_acc_slice_xy_level != accumulation_queue_->get_size())
		{
			auto new_fd = fd_;
			new_fd.depth = cd_.img_type == ImgType::Composite ? 12.0 : 4.0;
			try
			{
				accumulation_queue_.reset(new Queue(new_fd, cd_.img_acc_slice_xy_level.load(), "AccumulationQueueXY"));
			}
			catch (std::logic_error&)
			{
				accumulation_queue_.reset(nullptr);
			}
			if (!accumulation_queue_)
				std::cerr << "Error: can't allocate queue" << std::endl;
		}
	}
	else if (!queue_needed)
		accumulation_queue_.reset(nullptr);
	if (accumulation_queue_)
	{
		fn_vect_.push_back([=]() {
			if (accumulation_queue_)
			{
				if (!float_buffer_average_)
				{
					float *tmp = nullptr;
					cudaMalloc<float>(&tmp, accumulation_queue_->get_frame_desc().frame_size());
					float_buffer_average_.reset(tmp);
				}
				if (cd_.img_acc_slice_xy_enabled)
					accumulate_images(
						static_cast<float *>(accumulation_queue_->get_buffer()),
						float_buffer_average_.get(),
						accumulation_queue_->get_start_index(),
						accumulation_queue_->get_max_elts(),
						cd_.img_acc_slice_xy_level.load(),
						accumulation_queue_->get_frame_desc().frame_size() / sizeof(float),
						0);
				// TODO stabilize here
				accumulation_queue_->enqueue(gpu_float_buffer_, cudaMemcpyDeviceToDevice);
				if (cd_.img_acc_slice_xy_enabled)
					cudaMemcpy(gpu_float_buffer_, float_buffer_average_.get(), accumulation_queue_->get_frame_desc().frame_size(), cudaMemcpyDeviceToDevice);
			}
		});
	}
}
