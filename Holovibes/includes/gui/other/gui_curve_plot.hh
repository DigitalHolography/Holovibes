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
 * Widget wrapping for a QtChart. Used to display Chart/ROI computations. */
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

using namespace QtCharts;

namespace holovibes
{
	namespace gui
	{
		/*! \brief Widget wrapping for a QtChart. Used to display Chart/ROI computations. */
		class CurvePlot : public QWidget
		{
			Q_OBJECT

		public:
			/*! \brief CurvePlot constructor
			**
			** \param data_vect ConcurrentDeque containing chart values to be display
			** \param title title of the plot
			** \param width width of the plot in pixels
			** \param height height of the plot in pixels
			** \param parent Qt parent
			*/
			CurvePlot(ConcurrentDeque<ChartPoint>& data_vect,
				const size_t auto_scale_point_threshold,
				const QString title,
				const unsigned int width,
				const unsigned int height,
				QWidget* parent = nullptr);

			/*! \brief CurvePlot destructor */
			~CurvePlot();

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
			**
			** Copy the data from the holovibes Deque to a local vector. Then attach
			** it to line_series.
			*/
			void load_data_vector();

			/*! \brief Reajust the scale according to max and min values contained in deque.
			**
			** Look for current min and max values in the data Deque then adjust the chart
			** scale according to these values and replot.
			*/
			void auto_scale();

			/*! \brief Starts the chart drawing
			**
			** Starts timer_.
			*/
			void start();

			/*! \brief Stops the chart drawing
			**
			** Stop timer_ if active.
			*/
			void stop();

			/*! \brief Sets a new number of points for the chart
			**
			** Stops the drawing then change the number of points and resize the vector then
			** starts again.
			*/
			void set_points_nb(const unsigned int n);

			/*! Swtich between light and dark mode */
			void toggle_dark_mode(bool dark_mode);

			public slots:
			/*! \brief Updates the chart */
			void update();

			/*! \brief Change the curve ploted by changing curve_get_ */
			void change_curve(int curve_to_plot);

		private:
			/*! Data points on the chart */
			QLineSeries *line_series;
			/*! The chart itself */
			QChart *chart;
			/*! QtWidget used to display the chart on a window */
			QChartView *chart_view;

			/*! Reference to Deque containing Chart/ROI data */
			ConcurrentDeque<ChartPoint>& data_vect_;
			/*! Number of points to draw */
			unsigned int points_nb_;
			/*! QTimer used to draw every TIMER_FREQ milliseconds */
			QTimer timer_;
			/*! Ptr to function who get value of curve in tuple */
			std::function<double(const ChartPoint&)> curve_get_;
			/*! Local copy of data_vect data */
			std::vector<ChartPoint> chart_vector_;

			/*! Numbers of new points to wait before running auto scale */
			size_t auto_scale_point_threshold_;
			/*! Numbers of new points that already arrived since last auto_scale */
			size_t auto_scale_curr_points_;
		};
	}
}
