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

#include <filesystem>
#include <chrono>

#include <direct.h>

#include "recorder.hh"
#include "queue.hh"
#include "logger.hh"
#include "output_file_handler.hh"

# include "gui_group_box.hh"
# include "info_manager.hh"

namespace holovibes
{
	Recorder::Recorder(
		Queue& queue,
		const std::string& filepath,
		ComputeDescriptor& cd)
		: queue_(queue)
		, stop_requested_(false)
		, cd_(cd)
	{
		output_path_ = filepath;
	}

	Recorder::~Recorder()
	{
		// return to the .exe directory
		_chdir(execDir.c_str());
	}

	void Recorder::record()
	{
		// Flag to start the process of recording in the thread compute
		cd_.is_recording = true;
		queue_.clear();
		// Request enqueueing (or copying) frames for recording
		cd_.request_recorder_copy_frames = true;

		const unsigned int n_images = cd_.nb_frames_record;

		// While the request hasn't been completed
		while (cd_.request_recorder_copy_frames)
			continue;

		const size_t size = queue_.get_frame_size();
		char* buffer = new char[size];
		unsigned int cur_size = queue_.get_size();
		const unsigned int max_size = queue_.get_max_size();

		LOG_INFO(std::string("[RECORDER] started recording ") + std::to_string(n_images) + std::string(" frames"));

		const camera::FrameDescriptor& queue_fd = queue_.get_fd();

		camera::FrameDescriptor file_fd = queue_fd;
		file_fd.depth = queue_fd.depth == 6 ? 3 : queue_fd.depth;

		try
		{
			io_files::OutputFileHandler::create(output_path_, file_fd, n_images);
			io_files::OutputFileHandler::export_compute_settings(cd_);

			io_files::OutputFileHandler::write_header();

			gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::SAVING_THROUGHPUT, "Saving Throughput", "0 MB/s");
			size_t written_bytes = 0;

			// The for can be break for two reasons:
			// -> the recording is requested to stop
			// -> the thread compute terminated copying frames and the queue is empty
			// The thread computes has the tasks to copy the correct number of frames
			unsigned int nb_frames_written = 0;
			while (!stop_requested_ && (queue_.get_size() > 0 || !cd_.copy_frames_done))
			{
				auto start_time = std::chrono::steady_clock::now();

				while (queue_.get_size() == 0)
					std::this_thread::yield();

				cur_size = queue_.get_size();
				if (cur_size >= max_size - 1)
				{
					gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::RECORDING,
																"Recording", "Queue is full, data will be lost !");
				}
				else if (cur_size > (max_size * 0.8f))
				{
					gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::RECORDING,
																"Recording", "Queue is nearly full !");
				}
				else
				{
					gui::InfoManager::get_manager()->remove_info("Recording");
				}

				if (queue_fd.depth == 6)
				{
					// Record 48-bit color image into 24-bit color
					queue_.dequeue_48bit_to_24bit(buffer, cudaMemcpyDeviceToHost);
					written_bytes = io_files::OutputFileHandler::write_frame(buffer, size / 2);
				}
				else
				{
					// Normal recording
					queue_.dequeue(buffer, cudaMemcpyDeviceToHost);
					written_bytes = io_files::OutputFileHandler::write_frame(buffer, size);
				}

				auto end_time = std::chrono::steady_clock::now();
				auto elapsed = end_time - start_time;
				long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
				long long saving_rate = written_bytes / microseconds; // bytes / microsecond which is equal to MegaByte / second
				gui::InfoManager::get_manager()->update_info("Saving Throughput", std::to_string(saving_rate) + " MB/s");


				++nb_frames_written;
				emit value_change(nb_frames_written);
			}

			io_files::OutputFileHandler::write_footer();
			io_files::OutputFileHandler::close();

			if (nb_frames_written == n_images)
				LOG_INFO("[RECORDER] Record succeed !");
			else
			{
				LOG_INFO("[RECORDER] Record failed ! Number of frames written "
					+ std::to_string(nb_frames_written) + "/"
					+ std::to_string(n_images) + ".");
				LOG_INFO("[RECORDER] Increase output queue size !");
			}

			gui::InfoManager::get_manager()->remove_info("Recording");
			gui::InfoManager::get_manager()->remove_info("Saving Throughput");
		}
		catch (const io_files::FileException& e)
		{
			LOG_ERROR("[RECORDER] " + std::string(e.what()));
			io_files::OutputFileHandler::close();
		}

		delete[] buffer;

		// End of recording
		cd_.is_recording = false;
	}

	void Recorder::stop()
	{
		stop_requested_ = true;
	}
}
