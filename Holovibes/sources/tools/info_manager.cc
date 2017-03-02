#include "info_manager.hh"

namespace gui
{
	InfoManager* InfoManager::instance = nullptr;

	InfoManager* InfoManager::get_manager(gui::GroupBox *ui)
	{
		if (instance)
			return (instance);
		else if (ui)
			return ((InfoManager::instance = new InfoManager(ui)));
		else
			throw InfoManager::ManagerNotInstantiate();
	}

	void InfoManager::update_info_safe(const std::string& key, const std::string& value)
	{
		if (instance)
			instance->update_info(key, value);
	}

	void InfoManager::remove_info_safe(const std::string& key)
	{
		if (instance)
			instance->remove_info(key);
	}

	void InfoManager::stop_display()
	{
		if (instance)
			instance->stop_requested_ = true;
	}

	InfoManager::InfoManager(gui::GroupBox *ui)
		: ui_(ui)
		, progressBar_(ui->findChild<QProgressBar*>("infoProgressBar"))
		, stop_requested_(false)
	{
		progressBar_ = ui->findChild<QProgressBar*>("infoProgressBar");
		infoEdit_ = ui->findChild<QTextEdit*>("infoTextEdit");
		connect(this, SIGNAL(update_text(const QString)), infoEdit_, SLOT(setText(const QString)));
		this->start();
	}

	void InfoManager::run()
	{
		while (!stop_requested_)
		{
			draw();
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
	}

	InfoManager::~InfoManager()
	{
		if (instance)
			delete instance;
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

	void InfoManager::update_info(const std::string& key, const std::string& value)
	{
		infos_[key] = value;
	}

	void InfoManager::remove_info(const std::string& key)
	{
		infos_.erase(key);
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