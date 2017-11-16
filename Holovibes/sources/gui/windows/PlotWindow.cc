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

#include "PlotWindow.hh"
#include "concurrent_deque.hh"

namespace holovibes
{
	namespace gui
	{
		PlotWindow::PlotWindow(ConcurrentDeque<Tuple4f>& data_vect,
			const QString title,
			QWidget* parent)
			: QMainWindow(parent)
			, curve_plot_(data_vect, title, 580, 250, this)
		{
			ui.setupUi(this);
			this->show();

			curve_plot_.move(0, 30);
			unsigned int s = static_cast<unsigned int>(data_vect.size());
			if (s > 0)
				curve_plot_.resize_plot(s);

			QSpinBox *p = findChild<QSpinBox *>("PointsNbSpinBox");
			change_points_nb(p->value());
		}

		PlotWindow::~PlotWindow()
		{
		}

		void PlotWindow::closeEvent(QCloseEvent *event)
		{
			emit closed();
		}

		void PlotWindow::resizeEvent(QResizeEvent* e)
		{
			curve_plot_.resize(e->size().width() - 20, e->size().height() - 50);
		}

		void PlotWindow::start_drawing()
		{
			curve_plot_.start();
		}

		void PlotWindow::stop_drawing()
		{
			curve_plot_.stop();
		}

		void PlotWindow::auto_scale()
		{
			curve_plot_.auto_scale();
		}

		void PlotWindow::change_points_nb(int n)
		{
			curve_plot_.set_points_nb(n);
		}

		void PlotWindow::change_curve(int curve_to_plot)
		{
			curve_plot_.change_curve(curve_to_plot);
		}
	}
}

#include "Debug/moc_PlotWindow.cc"