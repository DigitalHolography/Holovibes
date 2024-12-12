/*! \file
 *
 * \brief Widget wrapping for a QtChart. Used to display Chart computations.
 */
#pragma once

#include <memory>
#include <vector>
#include <functional>

#include <QObject>
#include <QString>
#include <QTimer>

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>

#include "concurrent_deque.hh"
#include "chart_point.hh"

namespace holovibes::gui
{
/*! \class AnalysisCurvePlot
 *
 * \brief Widget wrapping for a QtChart. Used to display Chart computations.
 */
class AnalysisCurvePlot : public QWidget
{
    Q_OBJECT

  public:
    /*! \brief AnalysisCurvePlot constructor
     *
     * \param data_vect ConcurrentDeque containing chart values to be display
     * \param title title of the plot
     * \param width width of the plot in pixels
     * \param height height of the plot in pixels
     * \param parent Qt parent
     */
    AnalysisCurvePlot(ConcurrentDeque<ChartMeanVesselsPoint>& data_vect,
                      const size_t auto_scale_point_threshold,
                      const QString title,
                      const unsigned int width,
                      const unsigned int height,
                      QWidget* parent = nullptr);

    /*! \brief AnalysisCurvePlot destructor */
    ~AnalysisCurvePlot();

    /*! \brief Different curve options */
    enum CurveName
    {
        AVG_SIGNAL = 0,
        AVG_NOISE = 1,
        AVG_SIGNAL_DIV_AVG_NOISE = 2,
        LOG_AVG_SIGNAL_DIV_AVG_NOISE = 3,
        STD_SIGNAL = 4,
        STD_SIGNAL_DIV_AVG_NOISE = 5,
        STD_SIGNAL_DIV_AVG_SIGNAL = 6,
    };

    /*! \brief This property holds the recommended minimum size for the widget. */
    QSize minimumSizeHint() const override;
    /*! \brief This property holds the recommended size for the widget. */
    QSize sizeHint() const override;

    /*! \brief Change the maximum number of points in the curve */
    void resize_plot(const int size);

    /*! \brief Call when windows is resize */
    void resizeEvent(QResizeEvent* e) override;

    /*! \brief Load data from Deque
     *
     * Copy the data from the holovibes Deque to a local vector. Then attach
     * it to line_series.
     */
    void load_data_vector();

    /*! \brief Reajust the scale according to max and min values contained in deque.
     *
     * Look for current min and max values in the data Deque then adjust the chart
     * scale according to these values and replot.
     */
    void auto_scale();

    /*! \brief Starts the chart drawing
     *
     * Starts timer_.
     */
    void start();

    /*! \brief Stops the chart drawing
     *
     * Stop timer_ if active.
     */
    void stop();

    /*! \brief Sets a new number of points for the chart
     *
     * Stops the drawing then change the number of points and resize the vector then starts again.
     */
    void set_points_nb(const unsigned int n);

  public slots:
    /*! \brief Updates the chart */
    void update();

    /*! \brief Change the curve ploted by changing curve_get_ */
    void change_curve(int curve_to_plot);

  private:
    /*! \brief Data points on the chart */
    QLineSeries* line_series;
    /*! \brief The chart itself */
    QChart* chart;
    /*! \brief QtWidget used to display the chart on a window */
    QChartView* chart_view;

    /*! \brief Reference to Deque containing Chart data */
    ConcurrentDeque<ChartMeanVesselsPoint>& data_vect_;
    /*! \brief Number of points to draw */
    unsigned int points_nb_;
    /*! \brief QTimer used to draw every TIMER_FREQ milliseconds */
    QTimer timer_;
    /*! \brief Ptr to function who get value of curve in tuple */
    std::function<double(const ChartMeanVesselsPoint&)> curve_get_;
    /*! \brief Local copy of data_vect data */
    std::vector<ChartMeanVesselsPoint> chart_vector_;

    /*! \brief Numbers of new points to wait before running auto scale */
    size_t auto_scale_point_threshold_;
    /*! \brief Numbers of new points that already arrived since last auto_scale */
    size_t auto_scale_curr_points_;
};
} // namespace holovibes::gui
