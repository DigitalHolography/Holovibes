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

#define WIDTH 600
#define HEIGHT 300
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
			const QString title,
			const unsigned int width,
			const unsigned int height,
			QWidget* parent)
			: QWidget(parent)
			, data_vect_(data_vect)
			, plot_(QString::fromLocal8Bit(""), this)
			, curve_("First curve")
			, points_nb_(POINTS)
			, timer_(this)
			, curve_get_(curve_get_3)
		{
			this->setMinimumSize(width, height);
			plot_.setMinimumSize(width, height);
			plot_.setAxisScale(0, -5.0, 15.0, 2.0);
			plot_.setCanvasBackground(QColor(255, 255, 255));
			curve_.setPen(QColor(0, 0, 0));
			show();
			average_vector_.resize(points_nb_);
			connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
			timer_.start(TIMER_FREQ);
		}

		CurvePlot::~CurvePlot()
		{
		}

		void CurvePlot::change_curve(int curve_to_plot)
		{
			std::cout << curve_to_plot << std::endl;

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
			plot_.resize(QSize::QSize(size, 0));
			points_nb_ = size;
			average_vector_.resize(size);
		}

		void CurvePlot::resizeEvent(QResizeEvent* e)
		{
			plot_.resize(e->size());
		}

		void CurvePlot::load_data_vector()
		{
			QVector<QPointF> new_data;

			if (!data_vect_.empty())
			{
				size_t copied_elts_nb = data_vect_.fill_array(average_vector_, points_nb_);

				for (size_t i = 0; i < copied_elts_nb; ++i)
					new_data << QPointF(i, curve_get_(average_vector_[i]));
			}
			curve_.setSamples(new_data);
			curve_.attach(&plot_);
		}

		void CurvePlot::auto_scale()
		{
			std::vector<Tuple4f> tmp = average_vector_;

			//float curr = 0.0f;

			auto minmax = std::minmax_element(tmp.cbegin(),
				tmp.cend(),
				[&](const Tuple4f& lhs, const Tuple4f& rhs)
			{
				return curve_get_(lhs) < curve_get_(rhs);
			});

			plot_.setAxisScale(0,
				curve_get_(*(minmax.first)) - 1.0,
				curve_get_(*(minmax.second)) + 1.0,
				2.0);
			plot_.replot();
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
			points_nb_ = n;
			average_vector_.resize(n);
			start();
		}

		void CurvePlot::update()
		{
			load_data_vector();

			while (data_vect_.size() > points_nb_)
				data_vect_.pop_front();

			plot_.replot();
		}
	}
}
#undef WIDTH
#undef HEIGHT
#undef TIMER_FREQ
#undef POINTS

#include "moc_gui_curve_plot.cc"