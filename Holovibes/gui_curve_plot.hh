#ifndef GUI_CURVE_PLOT_HH
# define GUI_CURVE_PLOT_HH

# include <QtWidgets>
# include <qwt_plot.h>
# include <qwt_plot_curve.h>
# include <qwt_series_data.h>
# include <vector>
# include <tuple>
# include <iostream>

namespace gui
{
  class CurvePlot : public QWidget
  {
    Q_OBJECT

  public:
    CurvePlot(const std::vector<std::tuple<float, float, float>>& data_vect,
      QString title,
      QWidget* parent = 0);
    ~CurvePlot();
    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;

    void load_data_vector();

  public slots:
    void update();

  private:
    const std::vector<std::tuple<float, float, float>>& data_vect_;
    QwtPlot plot_;
    QwtPlotCurve curve_;
    QTimer timer_;
  };
}

#endif /* !GUI_CURVE_PLOT_HH */