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

#include <windows.h>
#include <iostream>

#include "thread_capture.hh"
#include "info_manager.hh"
#include "icamera.hh"
#include "queue.hh"

namespace holovibes
{
	ThreadCapture::ThreadCapture(
		camera::ICamera& camera,
		Queue& input,
		SquareInputMode mode)
		: IThreadInput()
		, camera_(camera)
		, queue_(input)
		, thread_(&ThreadCapture::thread_proc, this)
	{
		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::IMG_SOURCE, "ImgSource", camera_.get_name());
		queue_.set_square_input_mode(mode);
		auto fd = get_input_fd();
		std::string input_descriptor_info = std::to_string(fd.width)
			+ std::string("x")
			+ std::to_string(fd.height)
			+ std::string(" - ")
			+ std::to_string(static_cast<int>(fd.depth * 8))
			+ std::string("bit");
		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::INPUT_SOURCE, "InputFormat", input_descriptor_info);
	}

	ThreadCapture::~ThreadCapture()
	{
		stop_requested_ = true;

		while (!thread_.joinable())
			continue;
		thread_.join();
		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::IMG_SOURCE, "ImgSource", "None");
	}

	void ThreadCapture::thread_proc()
	{
		SetThreadPriority(thread_.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
		while (!stop_requested_)
		{
            auto frame = camera_.get_frame();
			queue_.enqueue(frame, cudaMemcpyHostToDevice);
		}
	}

	const camera::FrameDescriptor& ThreadCapture::get_input_fd() const
	{
		return camera_.get_fd();
	}

	const camera::FrameDescriptor& ThreadCapture::get_queue_fd() const
	{
		return queue_.get_fd();
	}
}
