#ifndef GUI_CURVE_PLOT_HH
# define GUI_CURVE_PLOT_HH

# include <QtWidgets>
# include <qwt_plot.h>
# include <qwt_plot_curve.h>
# include <qwt_series_data.h>
# include <vector>
# include <tuple>

namespace gui
{
  class CurvePlot : public QwtPlot
  {
  public:
    CurvePlot(QString title, QWidget* parent = 0);
    ~CurvePlot();
    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;

    void load_data_vector(const std::vector<std::tuple<float, float, float>>& vector);

  private:
    QwtPlotCurve curve_;
  };
}

#endif /* !GUI_CURVE_PLOT_HH */