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

#include "info_manager.hh"

namespace holovibes
{
	namespace gui
	{
		InfoManager *InfoManager::instance = nullptr;

		InfoManager::InfoManager(gui::GroupBox *ui) :
			delError(nullptr),
			flag(Null),
			ui_(ui),
			progressBar_(ui->findChild<QProgressBar*>("RecordProgressBar")),
			stop_requested_(false),
			infos_{
				{ { std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") },
				{ std::make_pair("", "") } } }
		{
			progressBar_ = ui->findChild<QProgressBar*>("RecordProgressBar");
			infoEdit_ = ui->findChild<QTextEdit*>("InfoTextEdit");
			connect(this, SIGNAL(update_text(const QString)), infoEdit_, SLOT(setText(const QString)));
			this->start();
		}

		InfoManager::~InfoManager()
		{
			if (instance)
				delete instance;
		}

		InfoManager *InfoManager::get_manager(gui::GroupBox *ui)
		{
			if (instance)
				return instance;
			else if (ui)
				return InfoManager::instance = new InfoManager(ui);
			else
				throw InfoManager::ManagerNotInstantiate();
		}

		void InfoManager::startDelError(const std::string& key)
		{
			if (!delError && flag == Null)
			{
				delError = new std::thread(&InfoManager::taskDelError,key);
				flag = Operating;
			}
		}

		void InfoManager::taskDelError(const std::string& key)
		{
			using namespace std::chrono_literals;
			std::this_thread::sleep_for(3s);

			InfoManager::remove_info(key);
		}

		std::thread* InfoManager::getDelErrorThread()
		{
			return delError;
		}

		void InfoManager::joinDelErrorThread()
		{
			delError->join();
			delete delError;
			delError = nullptr;
			flag = Null;
		}

		void InfoManager::insertInputSource(const int width, const int height, const int depth)
		{
			std::string output_descriptor_info =
				std::to_string(width) + std::string("x") + std::to_string(height) +
				std::string(" - ") +
				std::to_string(static_cast<int>(depth * 8)) + std::string("bit");
			InfoManager::get_manager()->insert_info(InfoManager::InfoType::OUTPUT_SOURCE, "", "_______________");
			InfoManager::get_manager()->insert_info(InfoManager::InfoType::OUTPUT_SOURCE, "OutputFormat", output_descriptor_info);
		}

		void InfoManager::update_info(const std::string& key, const std::string& value)
		{
			if (instance)
			{
				int i;
				for (i = 0; i < instance->infos_.size(); ++i)
				{
					if (instance->infos_[i].first == key)
					{
						instance->infos_[i].second = value;
						return;
					}
				}
				instance->infos_.push_back(std::make_pair(key, value));
			}
		}

		void InfoManager::remove_info(const std::string& key)
		{
			int i = 0;

			if (instance)
			{
				for (i = 0; i < instance->infos_.size(); ++i)
					if (instance->infos_[i].first == key)
						break;
				if (i != instance->infos_.size())
					instance->infos_.erase(instance->infos_.begin() + i);
				if (key == "Error" || key == "Info")
					instance->flag = Finish;
			}
		}

		void InfoManager::insert_info(const uint pos, const std::string& key, const std::string& value)
		{
			int i = 0;

			if (instance)
			{
				for (i = 0; i < instance->infos_.size(); ++i)
					if (instance->infos_[i].first == key)
						break;
				if (i == instance->infos_.size())
				{
					if (pos < instance->infos_.size())
						instance->infos_.insert(instance->infos_.begin() + pos, std::make_pair(key, value));
					else
						instance->infos_.push_back(std::make_pair(key, value));
				}
				else
					update_info(key, value);
			}
		}

		void InfoManager::stop_display()
		{
			if (instance)
				instance->stop_requested_ = true;
		}

		void InfoManager::run()
		{
			while (!instance->stop_requested_)
			{
				if (instance->flag == Finish)
				{
					instance->joinDelErrorThread();
				}
				draw();
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		}

		void InfoManager::draw()
		{
			if (instance)
			{
				std::string str = "";
				for (int i = 0; i < instance->infos_.size(); ++i)
					if (instance->infos_[i].first != "" && instance->infos_[i].second != "")
						str += instance->infos_[i].first + ((instance->infos_[i].first != "") ? ":\n  " : "") + instance->infos_[i].second + "\n";
				const QString qstr = str.c_str();
				emit update_text(qstr);
			}
		}

		void InfoManager::clear_infos()
		{
			if (instance)
			{
				instance->infos_.clear();
				insert_info(InfoType::IMG_SOURCE, "ImgSource", "None");
			}
		}

		QProgressBar* InfoManager::get_progress_bar()
		{
			return progressBar_;
		}
	}
}
