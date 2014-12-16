#ifndef GUI_CURVE_PLOT_HH
# define GUI_CURVE_PLOT_HH

# include <QtWidgets>
# include <QVector>
# include <qwt_plot.h>
# include <qwt_plot_curve.h>
# include <qwt_series_data.h>
# include <tuple>
# include <iostream>
# include <array>
# include <float.h>

# include "concurrent_deque.hh"

namespace gui
{
  class CurvePlot : public QWidget
  {
    Q_OBJECT

  public:
    CurvePlot(holovibes::ConcurrentDeque<std::tuple<float, float, float>>& data_vect,
      QString title,
      unsigned int width,
      unsigned int height,
      QWidget* parent = 0);
    ~CurvePlot();
    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;

    void resizeEvent(QResizeEvent* e) override;
    void load_data_vector();
    void auto_scale();
    void start();
    void stop();
    void set_points_nb(unsigned int n);

  public slots:
    void update();

  private:
    holovibes::ConcurrentDeque<std::tuple<float, float, float>>& data_vect_;
    QwtPlot plot_;
    QwtPlotCurve curve_;
    unsigned int points_nb_;
    QTimer timer_;
    std::vector<std::tuple<float, float, float>> average_vector_;
  };
}

#endif /* !GUI_CURVE_PLOT_HH */