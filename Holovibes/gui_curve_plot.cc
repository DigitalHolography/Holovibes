#include "gui_curve_plot.hh"

#define WIDTH 600
#define HEIGHT 300

namespace gui
{
  CurvePlot::CurvePlot(holovibes::ConcurrentDeque<std::tuple<float, float, float>>& data_vect,
    QString title,
    unsigned int width,
    unsigned int height,
    QWidget* parent)
    : QWidget(parent),
    data_vect_(data_vect),
    plot_(title, this),
    curve_("First curve"),
    timer_(this)
  {
    this->setMinimumSize(width, height);
    plot_.setMinimumSize(width, height);
    show();
    connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
    timer_.start(40);
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

  void CurvePlot::resizeEvent(QResizeEvent* e)
  {
    plot_.resize(e->size());
  }

  void CurvePlot::load_data_vector()
  {
    QPolygonF new_data;

    if (!data_vect_.empty())
    {
      size_t copied_elts_nb = data_vect_.fill_array(average_array_);

      for (size_t i = 0; i < copied_elts_nb; ++i)
        new_data << QPointF(i, std::get<2>(average_array_[i]));
    }

    curve_.setSamples(new_data);
    curve_.attach(&plot_);
  }

  void CurvePlot::update()
  {
    load_data_vector();

    while (data_vect_.size() > 100)
      data_vect_.pop_front();

    plot_.setAxisScale(0, -5.0, 15.0, 2.0);
    plot_.replot();
  }
}