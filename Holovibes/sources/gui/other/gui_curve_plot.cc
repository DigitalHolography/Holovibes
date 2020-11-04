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

#include "gui_curve_plot.hh"
#include "PlotWindow.hh"

#define WIDTH 600
#define HEIGHT 300
#define TOP_OFFSET 30
#define SIZE_OFFSET 20
#define TIMER_FREQ 40
#define POINTS 200

/*! \brief CURVE_GET is a macro generating curve_get_X function whoose return std::get<X> of tuple give 
**
** This macro is use to instance 4 function curve_get_{0-3} in order to create ptr on */
# define CURVE_GET(X) float curve_get_ ## X(const holovibes::Tuple4f& a) { return std::get<X>(a); }
CURVE_GET(0)
CURVE_GET(1)
CURVE_GET(2)
CURVE_GET(3)

namespace holovibes
{
	namespace gui
	{
		CurvePlot::CurvePlot(ConcurrentDeque<Tuple4f>& data_vect,
			const size_t auto_scale_point_threshold,
			const QString title,
			const unsigned int width,
			const unsigned int height,
			QWidget* parent)
			: QWidget(parent)
			, data_vect_(data_vect)
			, points_nb_(POINTS)
			, timer_(this)
			, curve_get_(curve_get_0)
			, auto_scale_point_threshold_(auto_scale_point_threshold)
			, auto_scale_curr_points_(0)
		{
			line_series = new QLineSeries();
			chart = new QChart();
			chart_view = new QChartView(chart, parent);

			chart->setTheme(QChart::ChartTheme::ChartThemeDark);
			line_series->setColor(QColor::fromRgb(255, 255, 255));

			chart->legend()->hide();
			chart->addSeries(line_series);
			chart->createDefaultAxes();
			chart->setTitle(title);

			chart_view->setRenderHint(QPainter::Antialiasing);
			chart_view->move(0, TOP_OFFSET);

			chart_vector_.resize(points_nb_);
			connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
			timer_.start(TIMER_FREQ);
		}

		CurvePlot::~CurvePlot()
		{
			delete line_series;
			delete chart;
			delete chart_view;
		}

		void CurvePlot::change_curve(int curve_to_plot)
		{
			switch (static_cast<CurvePlot::CurveName>(curve_to_plot))
			{
			case CurvePlot::CurveName::CURVE_SIGNAL:
				curve_get_ = curve_get_0;
				break;
			case CurvePlot::CurveName::CURVE_NOISE:
				curve_get_ = curve_get_1;
				break;
			case CurvePlot::CurveName::CURVE_LOG:
				curve_get_ = curve_get_2;
				break;
			case CurvePlot::CurveName::CURVE_LOG10:
				curve_get_ = curve_get_3;
				break;
			default:
				abort();
			}
		}

		QSize CurvePlot::minimumSizeHint() const
		{
			return QSize(WIDTH, HEIGHT);
		}

		QSize CurvePlot::sizeHint() const
		{
			return QSize(WIDTH, HEIGHT);
		}

		void CurvePlot::resize_plot(const int size)
		{
			points_nb_ = size;
			chart_vector_.resize(size);
			chart->axisX()->setMax(QVariant(points_nb_));
		}

		void CurvePlot::resizeEvent(QResizeEvent* e)
		{
			chart_view->resize(e->size().width() + SIZE_OFFSET, e->size().height() + SIZE_OFFSET);
		}

		void CurvePlot::load_data_vector()
		{
			QList<QPointF> new_data;

			if (!data_vect_.empty())
			{
				size_t copied_elts_nb = data_vect_.fill_array(chart_vector_, points_nb_);
				new_data.reserve(copied_elts_nb);

				++auto_scale_curr_points_;

				for (size_t i = 0; i < copied_elts_nb; ++i)
				{
					float x = i;
					float y = curve_get_(chart_vector_[i]);
					new_data.push_back(QPointF(x, y));
				}

				if (auto_scale_curr_points_ > auto_scale_point_threshold_)
				{
					auto_scale_curr_points_ -= auto_scale_point_threshold_;
					auto_scale();
				}
			}

			line_series->replace(new_data);
		}

		void CurvePlot::auto_scale()
		{
			std::vector<Tuple4f> tmp = chart_vector_;

			auto minmax = std::minmax_element(tmp.cbegin(),
				tmp.cend(),
				[&](const Tuple4f& lhs, const Tuple4f& rhs)
			{
				return curve_get_(lhs) < curve_get_(rhs);
			});

			double min = curve_get_(*(minmax.first));
			double max = curve_get_(*(minmax.second));
			double offset = (max - min) / 10.0;

			chart->axisY()->setRange(min - offset, max + offset);
		}

		void CurvePlot::start()
		{
			timer_.start(TIMER_FREQ);
		}

		void CurvePlot::stop()
		{
			if (timer_.isActive())
				timer_.stop();
		}

		void CurvePlot::set_points_nb(const unsigned int n)
		{
			stop();
			resize_plot(n);
			start();
		}

		void CurvePlot::update()
		{
			load_data_vector();
			while (data_vect_.size() > points_nb_)
				data_vect_.pop_front();
		}

		void CurvePlot::toggle_dark_mode(bool dark_mode)
		{
			auto theme = QChart::ChartTheme::ChartThemeLight;
			auto line_color = QColor::fromRgb(0, 0, 0);

			if (dark_mode)
			{
				theme = QChart::ChartTheme::ChartThemeDark;
				line_color = QColor::fromRgb(255, 255, 255);
			}

			chart->setTheme(theme);
			line_series->setColor(line_color);
		}
	}
}
#undef WIDTH
#undef HEIGHT
#undef TOP_OFFSET
#undef SIZE_OFFSET
#undef TIMER_FREQ
#undef POINTS