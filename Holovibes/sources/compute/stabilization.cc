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
#include "rect.hh"
#include "power_of_two.hh"
#include "cufft_handle.hh"
#include "nppi_functions.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "stabilization.cuh"
#include "icompute.hh"
#include "logger.hh"
#include "nppi.h"


using holovibes::compute::Stabilization;
using holovibes::FnVector;
using holovibes::cuda_tools::CufftHandle;
using holovibes::cuda_tools::UniquePtr;
using holovibes::CoreBuffers;


Stabilization::Stabilization(FnVector& fn_vect,
	const CoreBuffers& buffers,
	const camera::FrameDescriptor& fd,
	const holovibes::ComputeDescriptor& cd)
	: fn_vect_(fn_vect)
	, buffers_(buffers)
	, fd_(fd)
	, cd_(cd)
	, nppi_data_(fd.width, fd.height)
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
		auto zone = cd_.getStabilizationZone();
		if (!zone.area())
			return;
		nppi_data_.set_size(zone.width(), zone.height());
		if (accumulation_queue_->get_current_elts())
		{
			if (!convolution_.ensure_minimum_size(zone.area()))
				return;
			compute_correlation(buffers_.gpu_float_buffer_, float_buffer_average_);
		}
	});
}

// x and y are the 2 images to correlate, **NOT COORDINATES**
void Stabilization::compute_correlation(const float *x, const float *y)
{
	auto zone = cd_.getStabilizationZone();
	const uint size = zone.area();
	QPoint dimensions{ zone.width(), zone.height() };
	UniquePtr<float> selected_x(size);
	UniquePtr<float> selected_y(size);
	if (!selected_x || !selected_y)
		return;

	extract_frame(x, selected_x.get(), fd_.width, zone);
	extract_frame(y, selected_y.get(), fd_.width, zone);

	normalize_frame(selected_x.get(), zone.area());
	normalize_frame(selected_y.get(), zone.area());


	compute_convolution(selected_x.get(), selected_y.get(), convolution_.get());
	cudaStreamSynchronize(0);
}


void Stabilization::compute_convolution(const float* x, const float* y, float* out)
{
	auto zone = cd_.getStabilizationZone();
	const uint size = zone.area();

	CufftHandle plan2d_a(zone.height(), zone.width(), CUFFT_R2C);
	CufftHandle plan2d_b(zone.height(), zone.width(), CUFFT_R2C);
	CufftHandle plan2d_inverse(zone.height(), zone.width(), CUFFT_C2R);

	constexpr uint s = 64;
	convolution_float(
		x,
		y,
		out,
		zone.area(),
		plan2d_a,
		plan2d_b,
		plan2d_inverse);
}

void Stabilization::insert_extremums()
{
	fn_vect_.push_back([=]()
	{
		auto zone = cd_.getStabilizationZone();
		if (!zone.area())
			return;
		const auto frame_res = zone.area();
		if (convolution_.is_large_enough(frame_res))
		{
			float max = 0;
			int x = 0;
			int y = 0;
			
			cuda_tools::nppi_get_max_index(convolution_.get(), nppi_data_, &max, &x, &y);

			// (0, 0) is top left of image so we need to center it
			if (x > zone.width() / 2)
				x -= zone.width();
			if (y > zone.height() / 2)
				y -= zone.height();

			shift_x = -x;
			shift_y = -y;
		}
	});
}


void Stabilization::insert_stabilization()
{
	fn_vect_.push_back([=]()
	{
		if (!cd_.getStabilizationZone().area())
			return;
		if (!cd_.xy_stabilization_paused)
			complex_translation(buffers_.gpu_float_buffer_, fd_.width, fd_.height, shift_x, shift_y);
		else
		{
			shift_x = 0;
			shift_y = 0;
		}
	});
}

void Stabilization::insert_average_compute()
{
	bool queue_needed = cd_.img_acc_slice_xy_enabled || cd_.xy_stabilization_enabled;
	if (queue_needed)
	{
		if (!accumulation_queue_ || cd_.img_acc_slice_xy_level != accumulation_queue_->get_max_elts())
		{
			auto new_fd = fd_;
			new_fd.depth = cd_.img_type == ImgType::Composite ? 12 : 4;
			try
			{
				accumulation_queue_.reset(new Queue(new_fd, cd_.img_acc_slice_xy_level, "AccumulationQueueXY"));
			}
			catch (std::logic_error&)
			{
				accumulation_queue_.reset(nullptr);
			}
			if (!accumulation_queue_)
				LOG_ERROR("Can't allocate queue");
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
			accumulation_queue_->enqueue(buffers_.gpu_float_buffer_);
			if (cd_.img_acc_slice_xy_enabled)
				cudaMemcpy(buffers_.gpu_float_buffer_, float_buffer_average_, accumulation_queue_->get_frame_desc().frame_size(), cudaMemcpyDeviceToDevice);
		});
	}



	if (cd_.xy_stabilization_enabled && cd_.xy_stabilization_show_convolution)
	{
		// Visualization of convolution matrix
		fn_vect_.push_back([=]()
		{
			cudaStreamSynchronize(0);
			auto zone = cd_.getStabilizationZone();

			if (convolution_.is_large_enough(zone.area()))
			{
				gpu_resize(convolution_, buffers_.gpu_float_buffer_, { zone.width(), zone.height() }, { fd_.width, fd_.height });
				cudaStreamSynchronize(0);
			}
		});
	}
}


