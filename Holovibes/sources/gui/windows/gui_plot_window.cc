#include "gui_plot_window.hh"
#include "concurrent_deque.hh"

namespace gui
{
  PlotWindow::PlotWindow(holovibes::ConcurrentDeque<std::tuple<float, float, float, float>>& data_vect,
    const QString title,
    QWidget* parent)
    : QMainWindow(parent)
    , curve_plot_(data_vect, title, 580, 250, this)
  {
    ui.setupUi(this);
    this->show();

    curve_plot_.move(0, 30);
	unsigned int s = static_cast<unsigned int>(data_vect.size());
    if (s > 0)
      curve_plot_.resize_plot(s);

	QSpinBox *p = findChild<QSpinBox *>("PointsNbSpinBox");
	change_points_nb(p->value());
  }

  PlotWindow::~PlotWindow()
  {
  }

  void PlotWindow::closeEvent(QCloseEvent *event)
  {
    emit closed();
  }

  void PlotWindow::resizeEvent(QResizeEvent* e)
  {
    curve_plot_.resize(e->size().width() - 20, e->size().height() - 50);
  }

  void PlotWindow::start_drawing()
  {
    curve_plot_.start();
  }

  void PlotWindow::stop_drawing()
  {
    curve_plot_.stop();
  }

  void PlotWindow::auto_scale()
  {
    curve_plot_.auto_scale();
  }

  void PlotWindow::change_points_nb(int n)
  {
    curve_plot_.set_points_nb(n);
  }

  void PlotWindow::change_curve(int curve_to_plot)
  {
    curve_plot_.change_curve(curve_to_plot);
  }
}