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

#include "image_accumulation.hh"
#include "compute_descriptor.hh"
#include "rect.hh"
#include "power_of_two.hh"
#include "cufft_handle.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "icompute.hh"
#include "logger.hh"


using holovibes::compute::ImageAccumulation;
using holovibes::FnVector;
using holovibes::cuda_tools::CufftHandle;
using holovibes::cuda_tools::UniquePtr;
using holovibes::CoreBuffers;


ImageAccumulation::ImageAccumulation(FnVector& fn_vect,
	const CoreBuffers& buffers,
	const camera::FrameDescriptor& fd,
	const holovibes::ComputeDescriptor& cd)
	: fn_vect_(fn_vect)
	, buffers_(buffers)
	, fd_(fd)
	, cd_(cd)
{}

void ImageAccumulation::insert_image_accumulation()
{
	insert_average_compute();
	insert_float_buffer_overwrite();
}

void ImageAccumulation::insert_average_compute()
{
	bool queue_needed = cd_.img_acc_slice_xy_enabled;
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
				cudaMalloc<float>(&tmp, accumulation_queue_->get_fd().frame_size());
				float_buffer_average_.reset(tmp);
			}
			accumulate_images(
				static_cast<float *>(accumulation_queue_->get_buffer()),
				float_buffer_average_.get(),
				accumulation_queue_->get_start_index(),
				accumulation_queue_->get_max_elts(),
				cd_.img_acc_slice_xy_level,
				accumulation_queue_->get_fd().frame_size() / sizeof(float),
				0);
		});
	}
}

void ImageAccumulation::insert_float_buffer_overwrite()
{
	if (accumulation_queue_)
	{
		fn_vect_.push_back([=]()
		{
			accumulation_queue_->enqueue(buffers_.gpu_float_buffer_);
			if (cd_.img_acc_slice_xy_enabled)
				cudaMemcpy(buffers_.gpu_float_buffer_, float_buffer_average_, accumulation_queue_->get_fd().frame_size(), cudaMemcpyDeviceToDevice);
		});
	}
}
