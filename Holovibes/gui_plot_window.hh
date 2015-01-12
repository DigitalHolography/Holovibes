#ifndef GUI_PLOT_WINDOW_HH
# define GUI_PLOT_WINDOW_HH

# include <QMainWindow>
# include <QtWidgets>
# include "ui_plot_window.h"
# include "gui_curve_plot.hh"
# include "concurrent_deque.hh"

namespace gui
{
  /*! \class PlotWindow
  **
  ** Qt main window class containing a plot of computed average values.
  */
  class PlotWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    /*! \brief PlotWindow constructor
    **
    ** Create a PlotWindow and show it.
    **
    ** \param data_vect ConcurrentDeque containing average values to be display
    ** \param title title of the window
    ** \param parent Qt parent
    */
    PlotWindow(holovibes::ConcurrentDeque<std::tuple<float, float, float>>& data_vect,
      QString title,
      QWidget* parent = 0);
    ~PlotWindow();

    void resizeEvent(QResizeEvent* e) override;

    /*! \brief Starts drawing the chart/plot.
    ** 
    ** See CurvePLot::start()
    */
    void start_drawing();

    /*! \brief Stops drawing the chart/plot.
    **
    ** See CurvePlot::stop()
    */
    void stop_drawing();

  public slots:
    /*! \brief Reajust the scale according to max and min values contained in deque.
    ** 
    ** See CurvePLot::auto_scale()
    */
    void auto_scale();

    /*! \brief Change number of points of average displayed.
    **
    ** \param n number of points to display
    */
    void change_points_nb(int n);

  private:
    Ui::PlotWindow ui;

    /*! CurvePlot object */
    CurvePlot curve_plot_;
  };
}

#endif /* !GUI_PLOT_WINDOW_HH */