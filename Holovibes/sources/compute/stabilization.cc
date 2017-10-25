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
	insert_average_compute();
	if (cd_.xy_stabilization_enabled)
	{
		insert_correlation();
		insert_extremums();
		insert_stabilization();
	}
	insert_float_buffer_overwrite();
}

void Stabilization::insert_correlation()
{
	fn_vect_.push_back([=]()
	{
		gui::Rectangle zone = cd_.getStabilizationZone();
		auto frame_res = fd_.frame_res();
		if (accumulation_queue_->get_current_elts())
		{
			if (!convolution_.ensure_minimum_size(zone.area()))
				return;
			compute_correlation(gpu_float_buffer_, float_buffer_average_.get());
		}
	});
}

void Stabilization::compute_correlation(const float *x, const float *y)
{
	gui::Rectangle zone = cd_.getStabilizationZone();
	const uint size = zone.area();
	QPoint dimensions{ zone.width(), zone.height() };
	cuda_tools::UniquePtr<float> selected_x(size);
	cuda_tools::UniquePtr<float> selected_y(size);
	cuda_tools::UniquePtr<float> sum_x(size);
	cuda_tools::UniquePtr<float> sum_y(size);
	if (!selected_x || !selected_y)
		return;

	extract_frame(x, selected_x.get(), fd_.width, zone);
	extract_frame(y, selected_y.get(), fd_.width, zone);
	gpu_float_divide(selected_x.get(), zone.area(), 65536);
	gpu_float_divide(selected_y.get(), zone.area(), 65536);

	float test_frame_1[] = {
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0
	};
	float test_frame_2[] = {
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0
	};
	bool debug = false;
	float tmp1[64];
	float tmp2[64];
	float tmp3[64];
	float tmp4[64];
	float out[64];
	if (debug)
	{
		cudaMemcpy(selected_x.get(), test_frame_1, 64 * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(selected_y.get(), test_frame_2, 64 * 4, cudaMemcpyHostToDevice);
	}




	rotation_180(selected_y.get(), dimensions);
	cudaStreamSynchronize(0);

	compute_convolution(selected_x.get(), selected_y.get(), convolution_.get());
	sum_left_right(selected_x.get(), sum_x.get(), dimensions);
	sum_left_right(selected_y.get(), sum_y.get(), dimensions);
	cudaStreamSynchronize(0);
	if (debug)
		cudaMemcpy(tmp1, sum_x.get(), 64 * 4, cudaMemcpyDeviceToHost);
	if (debug)
		cudaMemcpy(tmp2, sum_y.get(), 64 * 4, cudaMemcpyDeviceToHost);
	if (debug)
		cudaMemcpy(tmp3, convolution_.get(), 64 * 4, cudaMemcpyDeviceToHost);

	sum_left_right_inplace(convolution_.get(), dimensions);
	cudaStreamSynchronize(0);
	if (debug)
		cudaMemcpy(tmp4, convolution_.get(), 64 * 4, cudaMemcpyDeviceToHost);

	compute_numerator(sum_x.get(), sum_y.get(), convolution_.get(), dimensions);
	sum_inplace_squared(selected_x.get(), dimensions);
	sum_inplace_squared(selected_y.get(), dimensions);
	cudaStreamSynchronize(0);

	sum_squared_minus_square_sum(selected_x.get(), sum_x.get(), dimensions);
	sum_squared_minus_square_sum(selected_y.get(), sum_y.get(), dimensions);
	cudaStreamSynchronize(0);

	correlation(convolution_.get(), selected_x.get(), selected_y.get(), dimensions);
	cudaStreamSynchronize(0);


	if (debug)
		cudaMemcpy(out, convolution_.get(), 64 * 4, cudaMemcpyDeviceToHost);
	if (debug)
		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 8; j++)
				std::cout << out[i * 8 + j];
			std::cout << std::endl;
		}
}


void Stabilization::compute_convolution(const float* x, const float* y, float* out)
{
	gui::Rectangle zone = cd_.getStabilizationZone();
	cufftHandle plan2d_a;
	cufftHandle plan2d_b;
	cufftHandle plan2d_inverse;

	cufftPlan2d(&plan2d_a, zone.height(), zone.height(), CUFFT_R2C);
	cufftPlan2d(&plan2d_b, zone.height(), zone.width(), CUFFT_R2C);
	cufftPlan2d(&plan2d_inverse, zone.height(), zone.width(), CUFFT_C2R);

	constexpr uint s = 64;
	convolution_float(
		x,
		y,
		out,
		zone.area(),
		plan2d_a,
		plan2d_b,
		plan2d_inverse);
	cufftDestroy(plan2d_a);
	cufftDestroy(plan2d_b);
	cufftDestroy(plan2d_inverse);
}

void Stabilization::insert_extremums()
{
	fn_vect_.push_back([=]()
	{
		gui::Rectangle zone = cd_.getStabilizationZone();
		const auto frame_res = zone.area();
		if (convolution_.is_large_enough(frame_res))
		{
			uint max = 0;
			gpu_extremums(convolution_.get(), frame_res, nullptr, nullptr, nullptr, &max);
			// x y: Coordinates of maximum of the correlation function
			int x = (max + zone.width() / 2) % zone.width();
			int y = (max + zone.height() / 2) / zone.width();
			if (x > zone.width() / 2)
				x -= zone.width();
			if (y > zone.height() / 2)
				y -= zone.height();
			//shift_x = (shift_x + x + fd.width) % fd.width;
			//shift_y = (shift_y + y + fd.height) % fd.height;

			shift_x = x;
			shift_y = y;

			std::cout << shift_x << ", " << shift_y << std::endl;
		}
	});
}


void Stabilization::insert_stabilization()
{
	// Stabilization
	fn_vect_.push_back([=]()
	{
		complex_translation(gpu_float_buffer_, fd_.width, fd_.height, shift_x, shift_y);
	});
}



void Stabilization::insert_average_compute()
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
				accumulation_queue_.reset(new Queue(new_fd, cd_.img_acc_slice_xy_level, "AccumulationQueueXY"));
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
			if (!float_buffer_average_)
			{
				float *tmp = nullptr;
				cudaMalloc<float>(&tmp, accumulation_queue_->get_frame_desc().frame_size());
				float_buffer_average_.reset(tmp);
			}
			accumulate_images(
				static_cast<float *>(accumulation_queue_->get_buffer()),
				float_buffer_average_.get(),
				accumulation_queue_->get_start_index(),
				accumulation_queue_->get_max_elts(),
				cd_.img_acc_slice_xy_level,
				accumulation_queue_->get_frame_desc().frame_size() / sizeof(float),
				0);
		});
	}
}


void Stabilization::insert_float_buffer_overwrite()
{
	if (accumulation_queue_)
	{
		fn_vect_.push_back([=]()
		{
			accumulation_queue_->enqueue(gpu_float_buffer_, cudaMemcpyDeviceToDevice);
			if (cd_.img_acc_slice_xy_enabled)
				cudaMemcpy(gpu_float_buffer_, float_buffer_average_.get(), accumulation_queue_->get_frame_desc().frame_size(), cudaMemcpyDeviceToDevice);
		});
	}



	if (cd_.xy_stabilization_enabled && cd_.xy_stabilization_show_convolution)
	{
		// Visualization of convolution matrix
		fn_vect_.push_back([=]()
		{
			cudaStreamSynchronize(0);
			gui::Rectangle zone = cd_.getStabilizationZone();
			if (convolution_.is_large_enough(zone.area()))
			{
				gpu_resize(convolution_.get(), gpu_float_buffer_, { zone.width(), zone.height() }, { fd_.width, fd_.height });
			}
		});
	}
}
