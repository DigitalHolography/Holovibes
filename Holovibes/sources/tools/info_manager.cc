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

namespace gui
{
	InfoManager *InfoManager::instance = nullptr;

	InfoManager::InfoManager(gui::GroupBox *ui)
		: ui_(ui)
		, progressBar_(ui->findChild<QProgressBar*>("RecordProgressBar"))
		, stop_requested_(false)
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

	void InfoManager::update_info(const std::string& key, const std::string& value)
	{
		if (instance)
			instance->infos_[key] = value;
	}

	void InfoManager::remove_info(const std::string& key)
	{
		if (instance)
			instance->infos_.erase(key);
	}

	void InfoManager::insert_info(uint pos, const std::string& key, const std::string& value)
	{
		//infos_.insert(pos, );
	}

	void InfoManager::stop_display()
	{
		if (instance)
			instance->stop_requested_ = true;
	}

	void InfoManager::run()
	{
		while (!stop_requested_)
		{
			draw();
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
	}

	void InfoManager::draw()
	{
		std::string str = "";
		auto ite = infos_.end();
		for (auto it = infos_.begin(); it != ite; ++it)
			str += it->first + ":\n  " + it->second + "\n";
		const QString qstr = str.c_str();
		emit update_text(qstr);
	}

	void InfoManager::clear_info()
	{
		infos_.clear();
	}

	QProgressBar* InfoManager::get_progress_bar()
	{
		return (progressBar_);
	}
}