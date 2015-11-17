#include "gui_curve_plot.hh"

#define WIDTH 600
#define HEIGHT 300
#define TIMER_FREQ 40
#define POINTS 200

namespace gui
{
  CurvePlot::CurvePlot(holovibes::ConcurrentDeque<std::tuple<float, float, float>>& data_vect,
    const QString title,
    const unsigned int width,
    const unsigned int height,
    QWidget* parent)
    : QWidget(parent)
    , data_vect_(data_vect)
    , plot_(title, this)
    , curve_("First curve")
    , points_nb_(POINTS)
    , timer_(this)
  {
    this->setMinimumSize(width, height);
    plot_.setMinimumSize(width, height);
    plot_.setAxisScale(0, -5.0, 15.0, 2.0);
    plot_.setCanvasBackground(QColor(255, 255, 255));
    curve_.setPen(QColor(0, 0, 0));
    show();
    average_vector_.resize(points_nb_);
    connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
    timer_.start(TIMER_FREQ);
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

  void CurvePlot::resize_plot(const int size)
  {
    plot_.resize(QSize::QSize(size, 0));
    points_nb_ = size;
    average_vector_.resize(size);
  }

  void CurvePlot::resizeEvent(QResizeEvent* e)
  {
    plot_.resize(e->size());
  }

  void CurvePlot::load_data_vector()
  {
    QVector<QPointF> new_data;

    if (!data_vect_.empty())
    {
      size_t copied_elts_nb = data_vect_.fill_array(average_vector_, points_nb_);

      for (size_t i = 0; i < copied_elts_nb; ++i)
        new_data << QPointF(i, std::get<2>(average_vector_[i]));
    }

    curve_.setSamples(new_data);
    curve_.attach(&plot_);
  }

  void CurvePlot::auto_scale()
  {
    using elt_t = std::tuple<float, float, float>;
    std::vector<elt_t> tmp = average_vector_;

    float curr = 0.0f;

    auto minmax = std::minmax_element(tmp.cbegin(),
      tmp.cend(),
      [](const elt_t& lhs, const elt_t& rhs)
    {
      return std::get<2>(lhs) < std::get<2>(rhs);
    });

    plot_.setAxisScale(0,
      std::get<2>(*(minmax.first)) - 1.0,
      std::get<2>(*(minmax.second)) + 1.0,
      2.0);
    plot_.replot();
  }

  void CurvePlot::start()
  {
    timer_.start(TIMER_FREQ);
  }

  void CurvePlot::stop()
  {
    if (timer_.isActive())
      timer_.stop();
  }

  void CurvePlot::set_points_nb(const unsigned int n)
  {
    stop();
    points_nb_ = n;
    average_vector_.resize(n);
    start();
  }

  void CurvePlot::update()
  {
    load_data_vector();

    while (data_vect_.size() > points_nb_)
      data_vect_.pop_front();

    plot_.replot();
  }
}

#undef WIDTH
#undef HEIGHT
#undef TIMER_FREQ
#undef POINTS