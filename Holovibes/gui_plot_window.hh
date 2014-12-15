#ifndef GUI_PLOT_WINDOW_HH
# define GUI_PLOT_WINDOW_HH

# include <QMainWindow>
# include <QtWidgets>
# include "ui_plot_window.h"
# include "gui_curve_plot.hh"
# include "concurrent_deque.hh"

namespace gui
{
  class PlotWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    PlotWindow(holovibes::ConcurrentDeque<std::tuple<float, float, float>>& data_vect,
      QString title,
      QWidget* parent = 0);
    ~PlotWindow();


    void resizeEvent(QResizeEvent* e) override;

  private:
    Ui::PlotWindow ui;

    CurvePlot curve_plot_;
  };
}

#endif /* !GUI_PLOT_WINDOW_HH */