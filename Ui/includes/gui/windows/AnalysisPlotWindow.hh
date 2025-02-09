/*! \file
 *
 * \brief Qt main window class containing a plot of computed chart values
 */
#pragma once

// Suppress all warnings in this auto-generated file
#pragma warning(push, 0)
#include "ui_AnalysisPlotWindow.h"
#pragma warning(pop)

#include "gui_analysis_curve_plot.hh"

/* Forward declarations. */
namespace holovibes
{
template <class T>
class ConcurrentDeque;
} // namespace holovibes

namespace holovibes::gui
{
/*! \class AnalysisPlotWindow
 *
 * \brief Qt main window class containing a plot of computed chart values.
 */
class AnalysisPlotWindow : public QMainWindow
{
    Q_OBJECT

  signals:
    void closed();

  public:
    /*! \brief AnalysisPlotWindow constructor
     *
     * Create a AnalysisPlotWindow and show it.
     *
     * \param data_vect ConcurrentDeque containing chart values to be display
     * \param title title of the window
     * \param parent Qt parent
     */
    AnalysisPlotWindow(ConcurrentDeque<ChartMeanVesselsPoint>& data_vect,
                       const size_t auto_scale_point_threshold,
                       const QString title,
                       QWidget* parent = nullptr);

    /*! \brief Destroy the AnalysisPlotWindow object. */
    ~AnalysisPlotWindow();

    /*! \brief Resize the AnalysisPlotWindow. */
    void resizeEvent(QResizeEvent* e) override;

    /*! \brief Starts drawing the chart/plot.
     *
     * See CurvePLot::start()
     */
    void start_drawing();

    /*! \brief Stops drawing the chart/plot.
     *
     * See CurvePlot::stop()
     */
    void stop_drawing();

  public slots:
    /*! \brief Reajust the scale according to max and min values contained in deque.
     *
     * See CurvePLot::auto_scale()
     */
    void auto_scale();

    /*! \brief Change number of points of chart displayed.
     *
     * \param n number of points to display
     */
    void change_points_nb(int n);

    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent* event) override;

    /*! \brief Ask curve_plot_ to change ploted curve */
    void change_curve(int curve_to_plot);

  private:
    Ui::AnalysisPlotWindow ui;

    /*! \brief CurvePlot object */
    AnalysisCurvePlot curve_plot_;
};
} // namespace holovibes::gui
