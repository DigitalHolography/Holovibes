#ifndef GUI_PLOT_WINDOW_HH
# define GUI_PLOT_WINDOW_HH

# include <QMainWindow>
# include "ui_plot_window.h"

namespace gui
{
  class PlotWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    PlotWindow(QWidget* parent = 0);
    ~PlotWindow();

  private:
    Ui::PlotWindow ui;
  };
}

#endif /* !GUI_PLOT_WINDOW_HH */