#include "gui_plot_window.hh"

namespace gui
{
  PlotWindow::PlotWindow(holovibes::ConcurrentDeque<std::tuple<float, float, float>>& data_vect,
    QString title,
    QWidget* parent)
    : QMainWindow(parent),
    curve_plot_(data_vect, title, 580, 250, this)
  {
    ui.setupUi(this);
    this->show();

    curve_plot_.move(0, 30);
  }

  PlotWindow::~PlotWindow()
  {
  }
}