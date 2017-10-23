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

#include <sstream>
#include "info_manager.hh"
#include "HoloWindow.hh"
#include "MainWindow.hh"

namespace holovibes
{
	namespace gui
	{
		std::atomic<bool> BasicOpenGLWindow::slicesAreLocked = true;

		HoloWindow::HoloWindow(QPoint p, QSize s, Queue& q, SharedPipe ic, MainWindow *main_window) :
			DirectWindow(p, s, q, KindOfView::Hologram),
			Ic(ic),
			main_window_(main_window)
		{}

		HoloWindow::~HoloWindow()
		{}

		std::shared_ptr<ICompute> HoloWindow::getPipe()
		{
			return Ic;
		}

		void	HoloWindow::initShaders()
		{
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/vertex.holo.glsl");
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/fragment.tex.glsl");
			Program->link();
			overlay_manager_.create_default();
		}

		void	HoloWindow::paintGL()
		{
			if (Cd->stft_view_enabled.load())
			{
				QPoint top_left;
				Cd->stftCursor(&top_left, AccessMode::Get);
				if (Cd && (Cd->x_accu_enabled || Cd->y_accu_enabled))
				{
					QPoint bottom_right(top_left);
					if (Cd->x_accu_enabled)
					{
						top_left.setX(Cd->x_accu_min_level);
						bottom_right.setX(Cd->x_accu_max_level);
					}
					if (Cd->y_accu_enabled)
					{
						top_left.setY(Cd->y_accu_min_level);
						bottom_right.setY(Cd->y_accu_max_level);
					}
					overlay_manager_.setDoubleCrossBuffer(top_left, bottom_right, QSize(Fd.width, Fd.height));
				}
				else
					overlay_manager_.setCrossBuffer(top_left, QSize(Fd.width, Fd.height));
			}

			DirectWindow::paintGL();
		}

		void	HoloWindow::mousePressEvent(QMouseEvent* e)
		{
			if (!Cd->stft_view_enabled.load())
				DirectWindow::mousePressEvent(e);
		}

		void	HoloWindow::mouseMoveEvent(QMouseEvent* e)
		{
			QPoint pos(e->x() * (Fd.width / static_cast<float>(width())),
				e->y() * (Fd.height / static_cast<float>(height())));
			mouse_position = pos;
			if (Cd->stft_view_enabled.load() && !slicesAreLocked.load())
				updateCursorPosition(pos);
			else if (!Cd->stft_view_enabled.load())
				DirectWindow::mouseMoveEvent(e);
		}

		void	HoloWindow::mouseReleaseEvent(QMouseEvent* e)
		{
			if (!Cd->stft_view_enabled.load())
				DirectWindow::mouseReleaseEvent(e);
			else if (e->button() == Qt::RightButton)
				resetTransform();
		}

		void	HoloWindow::wheelEvent(QWheelEvent *e)
		{
			if (!Cd->stft_view_enabled.load())
				BasicOpenGLWindow::wheelEvent(e);
		}

		void	HoloWindow::keyPressEvent(QKeyEvent* e)
		{
			DirectWindow::keyPressEvent(e);
			if (Cd->stft_view_enabled.load() && e->key() == Qt::Key::Key_Space)
			{
				if (!slicesAreLocked && Cd)
					last_clicked = mouse_position;
				else
					updateCursorPosition(mouse_position);
				slicesAreLocked.exchange(!slicesAreLocked.load());
				makeCurrent();
				if (slicesAreLocked.load())
				{
					setCursor(Qt::ArrowCursor);
				}
				else
				{
					setCursor(Qt::CrossCursor);
				}
			}
		}
		
		void	HoloWindow::focusInEvent(QFocusEvent *e)
		{
			QOpenGLWindow::focusInEvent(e);
			Cd->current_window.exchange(WindowKind::XYview);
			Cd->notify_observers();
		}

		void	HoloWindow::updateCursorPosition(QPoint pos)
		{
			std::stringstream ss;
			ss << "(Y,X) = (" << pos.y() << "," << pos.x() << ")";
			InfoManager::get_manager()->update_info("STFT Slice Cursor", ss.str());
			Cd->stftCursor(&pos, AccessMode::Set);
			// ---------------
			makeCurrent();
			if (!overlay_manager_.setCrossBuffer(pos, QSize(Fd.width, Fd.height)))
				return;
			if (!slicesAreLocked && Cd)
			{
				Cd->x_accu_min_level = std::min(mouse_position.x(), last_clicked.x());
				Cd->y_accu_min_level = std::min(mouse_position.y(), last_clicked.y());
				Cd->x_accu_max_level = std::max(mouse_position.x(), last_clicked.x());
				Cd->y_accu_max_level = std::max(mouse_position.y(), last_clicked.y());
				main_window_->notify();
			}
		}
	}
}
