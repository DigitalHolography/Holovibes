#include "info_manager.hh"

#include <iostream>

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

  void InfoManager::add_info(std::string& key, std::string& value)
  {
    infos_.insert(std::make_pair(key, value));
  }

  void InfoManager::update_info(std::string& key, std::string& value)
  {
    std::string str = "";

    infos_[key] = value;

      auto ite = infos_.end();
      for (auto it = infos_.begin(); it != ite; ++it)
        str += it->first + ": " + it->second;

      const QString qstr = str.c_str();
      emit update_text(qstr);
  }

  void InfoManager::remove_info(std::string& key)
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