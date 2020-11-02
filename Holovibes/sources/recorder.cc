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
#include "holo_file.hh"

# include "gui_group_box.hh"
# include "info_manager.hh"

namespace holovibes
{
	Recorder::Recorder(
		Queue& queue,
		const std::string& filepath)
		: queue_(queue)
		, file_()
		, stop_requested_(false)
	{
		output_path_ = filepath;
		file_.open(filepath, std::ios::binary | std::ios::trunc);
	}

	Recorder::~Recorder()
	{
		if (file_.is_open())
			file_.close();
		// return to the .exe directory
		_chdir(execDir.c_str());
	}

	void Recorder::record(const unsigned int n_images, const json& json_settings)
	{
		const size_t size = queue_.get_frame_size();
		char* buffer = new char[size]();
		size_t cur_size = queue_.get_current_elts();
		const size_t max_size = queue_.get_max_elts();

		LOG_INFO(std::string("[RECORDER] started recording ") + std::to_string(n_images) + std::string(" frames"));

		auto header = HoloFile::create_header(
			json_settings.value("pixel_bits", 8),
			json_settings.value("img_width", 1024),
			json_settings.value("img_height", 1024),
			n_images
		);
		file_.write((char*)(&header), sizeof(HoloFile::Header));

		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::SAVING_THROUGHPUT, "Saving Throughput", "0 MB/s");
		size_t written_bytes = 0;

		for (unsigned int i = 1; !stop_requested_ && i <= n_images; ++i)
		{
			auto start_time = std::chrono::steady_clock::now();

			while (queue_.get_current_elts() < 1)
				std::this_thread::yield();

			cur_size = queue_.get_current_elts();
			if (cur_size >= max_size - 1)
				gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::RECORDING, "Recording", "Queue is full, data will be lost !");
			else if (cur_size > (max_size * 0.8f))
				gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::RECORDING,  "Recording", "Queue is nearly full !");
			else
				gui::InfoManager::get_manager()->remove_info("Recording");


			if (queue_.get_fd().depth == 6)
			{
				// Record 48-bit color image into 24-bit color
				queue_.dequeue_48bit_to_24bit(buffer, cudaMemcpyDeviceToHost);
				file_.write(buffer, size/2);
				written_bytes = size / 2;
			}
			else
			{
				// Normal recording
				queue_.dequeue(buffer, cudaMemcpyDeviceToHost);
				cudaStreamSynchronize(queue_.get_stream());
				file_.write(buffer, size);
				written_bytes = size;
			}

			auto end_time = std::chrono::steady_clock::now();
			auto elapsed = end_time - start_time;
			long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
			long long saving_rate = written_bytes / microseconds; // bytes / microsecond which is equal to MegaByte / second
			gui::InfoManager::get_manager()->update_info("Saving Throughput", std::to_string(saving_rate) + " MB/s");

			emit value_change(i);
		}


		std::string json_str = json_settings.dump();
		file_.write(json_str.data(), json_str.size());
		file_.close();

		LOG_INFO("[RECORDER] record done !");
		gui::InfoManager::get_manager()->remove_info("Recording");
		gui::InfoManager::get_manager()->remove_info("Saving Throughput");
		delete[] buffer;
	}

	void Recorder::stop()
	{
		stop_requested_ = true;
	}

	void Recorder::createFilePath(const std::string folderName)
	{
		std::list<std::string> folderLevels;
		char* c_str = (char*)folderName.c_str();

		// Point to end of the string
		char* strPtr = &c_str[strlen(c_str) - 1];

		// Create a list of the folders which do not currently exist
		do
		{
			if (std::filesystem::exists(c_str))
				break;
			// Break off the last folder name, store in folderLevels list
			do
			{
				--strPtr;
			} while ((*strPtr != '\\') && (*strPtr != '/') && (strPtr >= c_str));
			folderLevels.push_front(std::string(strPtr + 1));
			strPtr[1] = 0;
		} while (strPtr >= c_str);

		if (folderLevels.empty())
			return;
		std::cout << folderLevels.back() << std::endl;
		folderLevels.pop_back();

		// Save the .exe directory before the record
		execDir = std::filesystem::current_path().string();
		if (_chdir(c_str))
		{
			throw std::exception("[RECORDER] error cannot _chdir directory");
		}

		// Create the folders iteratively
		std::string startPath = std::filesystem::current_path().string();
		for (std::list<std::string>::iterator it = folderLevels.begin(); it != folderLevels.end(); it++)
		{
			if (std::filesystem::create_directory(it->c_str()) == 0)
				throw std::exception("[RECORDER] error cannot create directory");

			_chdir(it->c_str());
		}
		_chdir(startPath.c_str());
	}

	bool Recorder::is_file_exist(const std::string& filepath)
	{
		std::ifstream ifs(filepath);
		return ifs.good();
	}
}
