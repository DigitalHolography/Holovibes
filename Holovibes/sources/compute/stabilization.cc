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
#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include <iostream>
#include <cufft.h>

using holovibes::compute::Stabilization;


Stabilization::Stabilization(FnVector& fn_vect,
	cuComplex* const& gpu_complex_frame,
	float* const& gpu_float_buffer,
	const camera::FrameDescriptor& fd,
	const holovibes::ComputeDescriptor& cd)
	: fn_vect_(fn_vect)
	, gpu_complex_frame_(gpu_complex_frame)
	, gpu_float_buffer_(gpu_float_buffer)
	, fd_(fd)
	, cd_(cd)
{}

void Stabilization::insert_pre_img_type()
{
	if (cd_.xy_stabilization_enabled.load())
	{
		insert_convolution();
		insert_extremums();
	}
}
void Stabilization::insert_post_img_type()
{
	if (cd_.xy_stabilization_enabled.load())
		insert_stabilization();
	insert_average();
}

void Stabilization::insert_convolution()
{
	fn_vect_.push_back([=]() {
		auto frame_res = fd_.frame_res();
		if (last_frame_)
		{
			if (!convolution_)
			{
				float *tmp = nullptr;
				cudaMalloc<float>(&tmp, frame_res * sizeof(float));
				convolution_.reset(tmp);
			}
			cudaStreamSynchronize(0);
			compute_convolution();
		}
		else
		{
			cufftComplex *tmp = nullptr;
			cudaMalloc<cufftComplex>(&tmp, frame_res * sizeof(cuComplex));
			last_frame_.reset(tmp);
		}
		cudaMemcpyAsync(last_frame_.get(), gpu_complex_frame_, frame_res * sizeof(cuComplex), cudaMemcpyDeviceToDevice, 0);
	});
}

void Stabilization::compute_convolution()
{
	cufftHandle plan2d_a;
	cufftHandle plan2d_b;

	cufftPlan2d(&plan2d_a, fd_.width, fd_.height, CUFFT_C2C); // C2C
	cufftPlan2d(&plan2d_b, fd_.width, fd_.height, CUFFT_C2C);

	shift_corners_complex(last_frame_.get(), fd_.width, fd_.height);
	cudaStreamSynchronize(0);

	convolution_operator(gpu_complex_frame_,
		last_frame_.get(),
		convolution_.get(),
		fd_.frame_res(),
		plan2d_a,
		plan2d_b);
	///*float test[2048];
	//cudaMemcpy(test, convolution_.get(), 2048 * 4, cudaMemcpyDeviceToHost);//
	cufftDestroy(plan2d_a);
	cufftDestroy(plan2d_b);
}

void Stabilization::insert_extremums()
{
	fn_vect_.push_back([=]() {
		if (convolution_)
		{
			const auto frame_res = fd_.frame_res();
			uint max = 0;
			gpu_extremums(convolution_.get(), frame_res, nullptr, nullptr, nullptr, &max);
			// x y: Coordinates of maximum of the correlation function
			int x = max % fd_.width;
			int y = max / fd_.width;
			if (x > fd_.width / 2)
				x -= fd_.width;
			if (y > fd_.height / 2)
				y -= fd_.height;
			std::cout << x << ", " << y << std::endl;
			//shift_x = (shift_x + x + fd.width) % fd.width;
			//shift_y = (shift_y + y + fd.height) % fd.height;
			shift_x = x;
			shift_y = y;
		}
	});
}


void Stabilization::insert_stabilization()
{
	// Visualization of convolution matrix
	if (cd_.xy_stabilization_show_convolution.load())
		fn_vect_.push_back([=]() {cudaMemcpy(gpu_float_buffer_, convolution_.get(), fd_.frame_res() * 4, cudaMemcpyDeviceToDevice); });
	else if (false)
		// Visualization of image
		fn_vect_.push_back([=]() { complex_translation(gpu_float_buffer_, fd_.width, fd_.height, shift_x, shift_y); });
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
