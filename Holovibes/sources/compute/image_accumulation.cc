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
#include "cuda_memory.cuh"


namespace holovibes
{
	using cuda_tools::CufftHandle;
	using cuda_tools::UniquePtr;
	namespace compute
	{
		ImageAccumulation::ImageAccumulation(FnVector& fn_vect,
			ImageAccEnv& image_acc_env,
			const CoreBuffers& buffers,
			const camera::FrameDescriptor& fd,
			const holovibes::ComputeDescriptor& cd)
			: fn_vect_(fn_vect)
			, image_acc_env_(image_acc_env)
			, buffers_(buffers)
			, fd_(fd)
			, cd_(cd)
		{}

		void ImageAccumulation::insert_image_accumulation()
		{
			insert_compute_average();
			insert_copy_accumulation_result();
		}

		void ImageAccumulation::allocate_accumulation_queue(
			std::unique_ptr<Queue>& gpu_accumulation_queue,
			cuda_tools::UniquePtr<float>& gpu_average_frame,
			const unsigned int accumulation_level,
			const camera::FrameDescriptor fd)
		{
			// If the queue is null or the level has changed
			if (!gpu_accumulation_queue
				|| accumulation_level != gpu_accumulation_queue->get_max_size())
			{
				gpu_accumulation_queue.reset(
					new Queue(fd, accumulation_level, "AccumulationQueue@allocate_accumalation_queue"));

				// accumulation queue successfully allocated
				if (!gpu_average_frame)
				{
					auto frame_size = gpu_accumulation_queue->get_fd().frame_size();
					gpu_average_frame.resize(frame_size);
				}
			}
		}

		void ImageAccumulation::allocate_accumulation_queues()
		{
			// XY view
			if (cd_.img_acc_slice_xy_enabled)
			{
				auto new_fd = fd_;
				new_fd.depth = cd_.img_type == ImgType::Composite ? 3 * sizeof(float) : sizeof(float);
				allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xy_queue,
					image_acc_env_.gpu_float_average_xy_frame,
					cd_.img_acc_slice_xy_level,
					new_fd);
			}
			else
				image_acc_env_.gpu_accumulation_xy_queue.reset(nullptr);

			// XZ view
			if (cd_.img_acc_slice_xz_enabled)
			{
				auto new_fd = fd_;
				new_fd.depth = sizeof(float);
				new_fd.height = cd_.nSize;
				allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xz_queue,
					image_acc_env_.gpu_float_average_xz_frame,
					cd_.img_acc_slice_xz_level,
					new_fd);
			}

			// YZ view
			if (cd_.img_acc_slice_yz_enabled)
			{
				auto new_fd = fd_;
				new_fd.depth = sizeof(float);
				new_fd.width = cd_.nSize;
				allocate_accumulation_queue(image_acc_env_.gpu_accumulation_yz_queue,
					image_acc_env_.gpu_float_average_yz_frame,
					cd_.img_acc_slice_yz_level,
					new_fd);
			}
		}

		void ImageAccumulation::compute_average(
			std::unique_ptr<Queue>& gpu_accumulation_queue,
			float* gpu_input_frame,
			float* gpu_ouput_average_frame,
			const unsigned int image_acc_level,
			const size_t frame_res)
		{
			if (gpu_accumulation_queue)
			{
				// Enqueue the computed frame in the accumulation queue
				gpu_accumulation_queue->enqueue(gpu_input_frame);

				// Compute the average and store it in the output frame
				accumulate_images(
					static_cast<float *>(gpu_accumulation_queue->get_data()),
					gpu_ouput_average_frame,
					gpu_accumulation_queue->get_size(),
					gpu_accumulation_queue->get_max_size(),
					image_acc_level,
					frame_res);
			}
		}

		void ImageAccumulation::insert_compute_average()
		{
			auto compute_average_lambda = [&]()
			{
				// XY view
				if (image_acc_env_.gpu_accumulation_xy_queue && cd_.img_acc_slice_xy_enabled)
					compute_average(image_acc_env_.gpu_accumulation_xy_queue,
						buffers_.gpu_float_buffer_.get(),
						image_acc_env_.gpu_float_average_xy_frame.get(),
						cd_.img_acc_slice_xy_level,
						buffers_.gpu_float_buffer_size_);

				// XZ view
				if (image_acc_env_.gpu_accumulation_xz_queue && cd_.img_acc_slice_xz_enabled)
					compute_average(image_acc_env_.gpu_accumulation_xz_queue,
						buffers_.gpu_float_cut_xz_.get(),
						image_acc_env_.gpu_float_average_xz_frame,
						cd_.img_acc_slice_xz_level,
						image_acc_env_.gpu_accumulation_xz_queue->get_fd().frame_res());

				// YZ view
				if (image_acc_env_.gpu_accumulation_yz_queue && cd_.img_acc_slice_yz_enabled)
					compute_average(image_acc_env_.gpu_accumulation_yz_queue,
						buffers_.gpu_float_cut_yz_.get(),
						image_acc_env_.gpu_float_average_yz_frame,
						cd_.img_acc_slice_yz_level,
						image_acc_env_.gpu_accumulation_yz_queue->get_fd().frame_res());
			};

			fn_vect_.emplace_back(compute_average_lambda);
		}

		void ImageAccumulation::insert_copy_accumulation_result()
		{
			auto copy_accumulation_result = [&]()
			{
				// XY view
				if (image_acc_env_.gpu_accumulation_xy_queue && cd_.img_acc_slice_xy_enabled)
					cudaXMemcpy(buffers_.gpu_float_buffer_,
								image_acc_env_.gpu_float_average_xy_frame,
								image_acc_env_.gpu_accumulation_xy_queue->get_fd().frame_size(),
								cudaMemcpyDeviceToDevice);

				// XZ view
				if (image_acc_env_.gpu_accumulation_xz_queue && cd_.img_acc_slice_xz_enabled)
					cudaXMemcpy(buffers_.gpu_float_cut_xz_,
								image_acc_env_.gpu_float_average_xz_frame,
								image_acc_env_.gpu_accumulation_xz_queue->get_fd().frame_size(),
								cudaMemcpyDeviceToDevice);

				// YZ view
				if (image_acc_env_.gpu_accumulation_yz_queue && cd_.img_acc_slice_yz_enabled)
					cudaXMemcpy(buffers_.gpu_float_cut_yz_,
								image_acc_env_.gpu_float_average_yz_frame,
								image_acc_env_.gpu_accumulation_yz_queue->get_fd().frame_size(),
								cudaMemcpyDeviceToDevice);
			};

			fn_vect_.emplace_back(copy_accumulation_result);
		}
	}
}
