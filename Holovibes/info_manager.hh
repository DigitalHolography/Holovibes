#pragma once

# include "gui_group_box.hh"
# include <QProgressBar>
# include <QTextBrowser>

# include <stdexcept>

namespace gui
{
  class InfoManager : public QObject
  {
    Q_OBJECT
  signals:
    void update_text(const QString);
  private:
    class ManagerNotInstantiate : std::exception
    {
      std::string& what() { return std::string("InfoManager is not instantiate, use InfoManager::get_manager with arg"); }
    };
    InfoManager(gui::GroupBox *ui);
    ~InfoManager();

  public:
    static InfoManager* get_manager(gui::GroupBox *ui = nullptr);

    void draw();
    void update_info(const std::string& key, const std::string& value);
    void remove_info(const std::string& key);
    void clear_info();

    QProgressBar* get_progress_bar();
  private:
    static InfoManager* instance;

    std::map<std::string, std::string>  infos_;
    gui::GroupBox*  ui_;
    QProgressBar*   progressBar_;
    QTextEdit*      infoEdit_;
  };
}