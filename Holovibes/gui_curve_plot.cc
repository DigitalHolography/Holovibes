#include "gui_curve_plot.hh"

namespace gui
{
  CurvePlot::CurvePlot(QString title, QWidget* parent)
    : QwtPlot(title, parent),
    curve_("First curve")
  {
    this->setMinimumSize(600, 300);
    show();
  }

  CurvePlot::~CurvePlot()
  {
  }

  QSize CurvePlot::minimumSizeHint() const
  {
    return QSize(500, 200);
  }

  QSize CurvePlot::sizeHint() const
  {
    return QSize(500, 200);
  }

  void CurvePlot::load_data_vector(const std::vector<std::tuple<float, float, float>>& vector)
  {
    QPolygonF new_data;
    static int frame_nb = 0;

    if (!vector.empty())
    {
      for (auto it = vector.begin(); it != vector.end(); ++it)
      {
        const std::tuple<float, float, float>& tuple = *it;

        new_data << QPointF(frame_nb, std::get<2>(tuple));
        ++frame_nb;
      }
    }

    curve_.setSamples(new_data);
    curve_.attach(this);
    replot();
  }
}