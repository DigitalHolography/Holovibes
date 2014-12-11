#include "gui_curve_plot.hh"

namespace gui
{
  CurvePlot::CurvePlot(QString title, QWidget* parent)
    : QwtPlot(title, parent)
  {

  }

  CurvePlot::~CurvePlot()
  {
  }
}