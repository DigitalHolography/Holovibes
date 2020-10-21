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

#include <cuda_runtime.h>

#include "info_manager.hh"
#include "tools.hh"

namespace holovibes
{
	namespace gui
	{
		using MutexGuard = std::lock_guard<std::recursive_mutex>;

		InfoManager::InfoManager(gui::GroupBox *ui) :
			delError(nullptr),
			flag(ThreadState::Null),
			ui_(ui),
			progressBar_(ui->findChild<QProgressBar*>("ExportProgressBar")),
			infoEdit_(ui->findChild<QTextEdit*>("InfoTextEdit")),
			stop_requested_(false)
		{
			connect(this, SIGNAL(update_text(const QString)), infoEdit_, SLOT(setText(const QString)));
			this->start();
		}

		InfoManager *InfoManager::get_manager(gui::GroupBox *ui)
		{
			static InfoManager* instance = nullptr;
			if (instance)
				return instance;
			else if (ui)
				return instance = new InfoManager(ui);
			else
				throw InfoManager::ManagerNotInstantiate();
		}

		void InfoManager::startDelError(const std::string& key)
		{
			MutexGuard mGuard(mutex_);

			if (!delError && flag == ThreadState::Null)
			{
				delError = new std::thread(&InfoManager::taskDelError, this, key);
				flag = ThreadState::Operating;
			}
		}

		void InfoManager::taskDelError(const std::string& key) // Private
		{
			using namespace std::chrono_literals;
			std::this_thread::sleep_for(3s);
			remove_info(key);
		}

		void InfoManager::joinDelErrorThread() // Private
		{
			delError->join();
			delete delError;
			delError = nullptr;
			flag = ThreadState::Null;
		}

		void InfoManager::insertFrameDescriptorInfo(const camera::FrameDescriptor& fd, InfoManager::InfoType infotype, std::string name)
		{
			std::string fd_info = std::to_string(fd.width) + "x" + std::to_string(fd.height) +
								   " - " + std::to_string(fd.depth * 8) + "bit";
			insert_info(infotype, name, fd_info);
		}

		void InfoManager::update_info(const std::string& key, const std::string& value)
		{
			MutexGuard mGuard(mutex_);

			for (int i = 0; i < infos_.size(); ++i)
			{
				if (infos_[i].first == key)
				{
					infos_[i].second = value;
					return;
				}
			}
			infos_.push_back(std::make_pair(key, value));
		}

		void InfoManager::remove_info(const std::string& key)
		{
			MutexGuard mGuard(mutex_);

			for (int i = 0; i < infos_.size(); ++i)
				if (infos_[i].first == key) {
					infos_.erase(infos_.begin() + i);
					break;
				}

			if (key == "Error" || key == "Info")
				flag = ThreadState::Finish;
		}

		void InfoManager::insert_info(const uint pos, const std::string& key, const std::string& value)
		{
			MutexGuard mGuard(mutex_);

			for (int i = 0; i < infos_.size(); ++i)
				if (infos_[i].first == key)
					return update_info(key, value);

			if (pos < infos_.size())
				infos_.insert(infos_.begin() + pos, std::make_pair(key, value));
			else
				infos_.push_back(std::make_pair(key, value));
		}

		void InfoManager::stop_display()
		{
			MutexGuard mGuard(mutex_);

			stop_requested_ = true;
		}

		void InfoManager::run() // Private
		{
			while (!stop_requested_)
			{
				if (flag == ThreadState::Finish)
					joinDelErrorThread();
				draw();
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		}

		void InfoManager::draw() // Private
		{
			std::string str;
			for (int i = 0; i < infos_.size(); ++i) {
				if (infos_[i].first != "")
					str += infos_[i].first + ":\n ";
				str += infos_[i].second + "\n";
			}
			size_t free, total;
			cudaMemGetInfo(&free, &total);
			str += "GPU memory:\n" +
				engineering_notation(free, 2) + "B free,\n" +
				engineering_notation(total, 2) + "B total";
			emit update_text(str.c_str());
		}

		void InfoManager::clear_infos()
		{
			MutexGuard mGuard(mutex_);

			infos_.clear();
			insert_info(InfoType::IMG_SOURCE, "ImgSource", "None");
		}

		QProgressBar* InfoManager::get_progress_bar()
		{
			return progressBar_;
		}
	}
}