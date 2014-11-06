#ifndef WINDOW_HH_
# define WINDOW_HH_

# include <QMainWindow>
# include "ui_main_window.h"

class Window : public QMainWindow
{
  Q_OBJECT

public:
  Window(QWidget *parent = 0);
  ~Window();

private:
  Ui::MainWindow ui;
};

#endif WINDOW_HH_