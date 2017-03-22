/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
 * 
 * Qt main window class containing a plot of computed average values  */
#pragma once

# include <QMainWindow>
# include <QtWidgets>

# include "ui_PlotWindow.h"
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