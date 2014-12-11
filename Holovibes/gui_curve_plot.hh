#ifndef GUI_CURVE_PLOT_HH
# define GUI_CURVE_PLOT_HH

# include <QtWidgets>
# include <qwt_plot.h>

namespace gui
{
  class CurvePlot : public QwtPlot
  {
  public:
    CurvePlot(QString title, QWidget* parent = 0);
    ~CurvePlot();
  };
}

#endif /* !GUI_CURVE_PLOT_HH */