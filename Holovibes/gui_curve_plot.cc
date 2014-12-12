#include "gui_curve_plot.hh"

#define WIDTH 600
#define HEIGHT 300

namespace gui
{
  CurvePlot::CurvePlot(std::deque<std::tuple<float, float, float>>& data_vect,
    QString title, 
    QWidget* parent)
    : QWidget(parent),
    data_vect_(data_vect),
    plot_(title, this),
    curve_("First curve"),
    timer_(this)
  {
    this->setMinimumSize(WIDTH, HEIGHT);
    plot_.setMinimumSize(WIDTH, HEIGHT);
    show();
    connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
    timer_.start(100);
  }

  CurvePlot::~CurvePlot()
  {
  }

  QSize CurvePlot::minimumSizeHint() const
  {
    return QSize(WIDTH, HEIGHT);
  }

  QSize CurvePlot::sizeHint() const
  {
    return QSize(WIDTH, HEIGHT);
  }

  void CurvePlot::load_data_vector()
  {
    QPolygonF new_data;
    unsigned int i = 0;

    if (!data_vect_.empty())
    {
      for (auto it = data_vect_.rbegin(); it != data_vect_.rend(); ++it)
      {
        std::tuple<float, float, float> tuple = *it;
        new_data << QPointF(i, std::get<2>(tuple));
        ++i;
      }
    }

    curve_.setSamples(new_data);
    curve_.attach(&plot_);
  }

  void CurvePlot::update()
  {
    load_data_vector();

    while (data_vect_.size() > 100)
    {
      data_vect_.pop_front();
    }

    plot_.replot();
  }
}