#pragma once

# include "gui_group_box.hh"
# include <QProgressBar>
# include <QTextBrowser>
# include <QThread>
#include <thread>
#include <chrono>

# include <stdexcept>

namespace gui
{
  /*! \brief InfoManager is a singleton use to control info printed in infoLayout 
  **
  ** You can use it from anywere, using update/remove info to print what you want
  ** or get  a progress_bar to inform your progression */
  class InfoManager : public QThread
  {
    Q_OBJECT
  signals :
    /*! Inform infoEdit_ about new text to display*/
    void update_text(const QString);
  private:
    /*! Throwed if try instance InfoManager more once*/
    class ManagerNotInstantiate : std::exception
    {
      std::string& what() { return std::string("InfoManager is not instantiate, use InfoManager::get_manager with arg"); }
    };
    /*! ctr */
    InfoManager(gui::GroupBox *ui);
    /*! dtr */
    ~InfoManager();

	void run() override;
  public:
    /*! Get the singleton, it's creat on first call 
    ** \param ui must containt infoProgressBar and infoTextEdit in child*/
    static InfoManager* get_manager(gui::GroupBox *ui = nullptr);
    /*! Call update_info only if singleton is instantiate
    ** Use it if your are before InfoManager instantiate, or if your are possibly in CLI mode */
    static void update_info_safe(const std::string& key, const std::string& value);
    /*! Call remove_info only if singleton is instantiate
    ** Use it if your are before InfoManager instantiate, or if your are possibly in CLI mode */
    static void remove_info_safe(const std::string& key);

	/*! Stop to refresh the info_panel display*/
	static void stop_display();

    /*! Draw all current information */
    void draw();
    /*! Add your information until is remove, and call draw()
    ** \param key is where you can access to your information
    ** \param value is your information linked to key */
    void update_info(const std::string& key, const std::string& value);
    /*! Remove information, and call draw */
    void remove_info(const std::string& key);
    /*! Remove all information, and call draw */
    void clear_info();
    /*! Return progress_bar, you can use it as you want */
    QProgressBar* get_progress_bar();
  private:
    /*! The singleton*/
    static InfoManager* instance;

    /*! Store all informations */
    std::map<std::string, std::string>  infos_;
    /*! ui where find infoProgressBar and infoTextEdit */
    gui::GroupBox*  ui_;
    /*! infoProgressBar, you can get it with get_progress_bar() */
    QProgressBar*   progressBar_;
    /*! infoTextEdit, use to display informations*/
    QTextEdit*      infoEdit_;

	bool stop_requested_;
  };
}