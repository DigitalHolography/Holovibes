#ifndef MAIN_WINDOW_HH_
# define MAIN_WINDOW_HH_

# include <QMainWindow>
# include "ui_main_window.h"

namespace holovibes
{
  class MainWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

  private:
    Ui::MainWindow ui;
  };
}

#endif /* !MAIN_WINDOW_HH_ */