/*! \file gui_plot_window.hh
 * 
 * Qt main window class containing a plot of computed average values.
 */
#pragma once

# include <QMainWindow>
# include <QtWidgets>

# include "ui_plot_window.h"
# include "gui_curve_plot.hh"

/* Forward declarations. */
namespace holovibes
{
  template <class T>
  class ConcurrentDeque;
}

namespace gui
{
  /*! \brief Qt main window class containing a plot of computed average values. */
  class PlotWindow : public QMainWindow
  {
    Q_OBJECT

  signals :
    void closed();

  public:
    /*! \brief PlotWindow constructor
    **
    ** Create a PlotWindow and show it.
    **
    ** \param data_vect ConcurrentDeque containing average values to be display
    ** \param title title of the window
    ** \param parent Qt parent
    */
    PlotWindow(holovibes::ConcurrentDeque<std::tuple<float, float, float, float>>& data_vect,
      const QString title,
      QWidget* parent = nullptr);

    /*! \brief Destroy the plotwindow object. */
    ~PlotWindow();

    /*! \brief Resize the plotwindow. */
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

    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent *event);

    /*! Ask curve_plot_ to change ploted curve */
    void change_curve(int curve_to_plot);

  private:
    Ui::PlotWindow ui;

    /*! CurvePlot object */
    CurvePlot curve_plot_;
  };
}