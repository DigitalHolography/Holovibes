#include "gui_plot_window.hh"

namespace gui
{
  PlotWindow::PlotWindow(QWidget* parent)
    : QMainWindow(parent)
  {
    ui.setupUi(this);
    this->show();
  }

  PlotWindow::~PlotWindow()
  {
  }
}