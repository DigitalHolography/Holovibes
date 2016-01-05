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

  InfoManager::InfoManager(gui::GroupBox *ui) : ui_(ui), progressBar_(ui->findChild<QProgressBar*>("infoProgressBar"))
  {
    progressBar_ = ui->findChild<QProgressBar*>("infoProgressBar");
    infoEdit_ = ui->findChild<QTextEdit*>("infoTextEdit");
    connect(this, SIGNAL(update_text(const QString)), infoEdit_, SLOT(setText(const QString)));
  }

  InfoManager::~InfoManager() {}

  void InfoManager::draw()
  {
    std::string str = "";

    auto ite = infos_.end();
    for (auto it = infos_.begin(); it != ite; ++it)
      str += it->first + ": " + it->second + "\n";

    const QString qstr = str.c_str();
    emit update_text(qstr);
  }

  void InfoManager::update_info(const std::string& key, const std::string& value)
  {
    infos_[key] = value;
    draw();
  }

  void InfoManager::remove_info(const std::string& key)
  {
    infos_.erase(key);
    draw();
  }

  void InfoManager::clear_info()
  {
    infos_.clear();
    draw();
  }

  QProgressBar* InfoManager::get_progress_bar()
  {
    return (progressBar_);
  }
}