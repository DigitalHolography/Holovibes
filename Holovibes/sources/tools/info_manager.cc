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

#include <thread>
#include <chrono>
#include "info_manager.hh"

namespace holovibes
{
	namespace gui
	{
		InfoManager *InfoManager::instance = nullptr;

		InfoManager::InfoManager(gui::GroupBox *ui) :
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
				return (instance);
			else if (ui)
				return ((InfoManager::instance = new InfoManager(ui)));
			else
				throw InfoManager::ManagerNotInstantiate();
		}

		void InfoManager::insertInputSource(int width, int height, int depth)
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
				auto it = std::find_if(instance->infos_.begin(), instance->infos_.end(),
					[key](const std::pair<std::string, std::string>& element) { return element.first == key; });
				if (it != instance->infos_.end())
					it->second = value;
				else
					instance->infos_.push_back(std::make_pair(key, value));
			}
		}

		void InfoManager::remove_info(const std::string& key)
		{
			if (instance)
			{
				auto it = std::find_if(instance->infos_.begin(), instance->infos_.end(),
					[key](const std::pair<std::string, std::string>& element) { return element.first == key; });
				if (it != instance->infos_.end())
					instance->infos_.erase(it);
			}
		}

		void InfoManager::insert_info(uint pos, const std::string& key, const std::string& value)
		{
			if (instance)
			{
				auto vector_iterator = std::find_if(instance->infos_.begin(), instance->infos_.end(),
					[key](const std::pair<std::string, std::string>& element) { return element.first == key; });
				if (vector_iterator == instance->infos_.end())
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

		void InfoManager::clear_info()
		{
			if (instance)
			{
				instance->infos_.clear();
				insert_info(InfoType::IMG_SOURCE, "ImgSource", "None");
			}
		}

		QProgressBar* InfoManager::get_progress_bar()
		{
			return (progressBar_);
		}
	}
}
