#pragma once

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
  /*! \brief Widget containing a QwtPlot. Used to display average/ROI computations. */
  class CurvePlot : public QWidget
  {
    Q_OBJECT

  public:
    /*! \brief CurvePlot constructor
    **
    ** \param data_vect ConcurrentDeque containing average values to be display
    ** \param title title of the plot
    ** \param width width of the plot in pixels
    ** \param height height of the plot in pixels
    ** \param parent Qt parent
    */
    CurvePlot(holovibes::ConcurrentDeque<std::tuple<float, float, float>>& data_vect,
      const QString title,
      const unsigned int width,
      const unsigned int height,
      QWidget* parent = nullptr);

    /*! \brief CurvePlot destructor */
    ~CurvePlot();

    /*! \brief This property holds the recommended minimum size for the widget. */
    QSize minimumSizeHint() const override;
    /*! \brief This property holds the recommended size for the widget. */
    QSize sizeHint() const override;

    /*! \brief Change the maximum number of points in the curve */
    void resize_plot(const int size);

    /*! \brief Call when windows is resize */
    void resizeEvent(QResizeEvent* e) override;

    /*! \brief Load data from Deque
    **
    ** Copy the data from the holovibes Deque to a local vector. Then attach
    ** it to the curve.
    */
    void load_data_vector();

    /*! \brief Reajust the scale according to max and min values contained in deque.
    **
    ** Look for current min and max values in the data Deque then adjust the plot
    ** scale according to these values and replot.
    */
    void auto_scale();

    /*! \brief Starts the plot drawing
    **
    ** Starts timer_.
    */
    void start();

    /*! \brief Stops the plot drawing
    **
    ** Stop timer_ if active.
    */
    void stop();

    /*! \brief Sets a new number of points for the plot
    **
    ** Stops the drawing then change the number of points and resize the vector then
    ** starts again.
    */
    void set_points_nb(const unsigned int n);

    public slots:
    /*! \brief Updates the plot */
    void update();

  private:
    /*! Reference to Deque containing average/ROI data */
    holovibes::ConcurrentDeque<std::tuple<float, float, float>>& data_vect_;
    /*! QwtPlot */
    QwtPlot plot_;
    /*! Plot's curve */
    QwtPlotCurve curve_;
    /*! Number of points to draw */
    unsigned int points_nb_;
    /*! QTimer used to draw every TIMER_FREQ milliseconds */
    QTimer timer_;
    /*! Local copy of data_vect data */
    std::vector<std::tuple<float, float, float>> average_vector_;
  };
}