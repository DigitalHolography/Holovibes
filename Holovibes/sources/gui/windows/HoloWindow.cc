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

namespace holovibes
{
	namespace gui
	{
		std::atomic<bool> BasicOpenGLWindow::slicesAreLocked = true;

		HoloWindow::HoloWindow(QPoint p, QSize s, Queue& q, SharedPipe ic) :
			DirectWindow(p, s, q, KindOfView::Hologram),
			Ic(ic)
		{}

		HoloWindow::~HoloWindow()
		{}

		void	HoloWindow::initShaders()
		{
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/vertex.holo.glsl");
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/fragment.tex.glsl");
			Program->link();
			Overlay.initShaderProgram();
		}

		void	HoloWindow::paintGL()
		{
			DirectWindow::paintGL();
			// ---------------
			if (Cd->stft_view_enabled.load())
			{
				Overlay.drawCross(0, 4);
			}
			// ---------------
			Vao.release();
		}

		void	HoloWindow::mousePressEvent(QMouseEvent* e)
		{
			if (!Cd->stft_view_enabled.load())
				DirectWindow::mousePressEvent(e);
		}

		void	HoloWindow::mouseMoveEvent(QMouseEvent* e)
		{
			if (!Cd->stft_view_enabled.load())
				DirectWindow::mouseMoveEvent(e);
			else if (Cd->stft_view_enabled.load() && !slicesAreLocked.load())
				updateCursorPosition(QPoint(
					e->x() * (Fd.width / static_cast<float>(width())),
					e->y() * (Fd.height / static_cast<float>(height()))));
		}

		void	HoloWindow::mouseReleaseEvent(QMouseEvent* e)
		{
			if (!Cd->stft_view_enabled.load())
			{
				DirectWindow::mouseReleaseEvent(e);
				if (e->button() == Qt::LeftButton)
				{
					if (Overlay.getConstZone().topLeft() !=
						Overlay.getConstZone().bottomRight() && 
						Overlay.getKind() != Zoom)
					{
						Rectangle texZone = Overlay.getTexZone(height(), Fd.width);
						if (Overlay.getKind() == Filter2D)
						{
							Cd->stftRoiZone(texZone, AccessMode::Set);
							Ic->request_filter2D_roi_update();
							Ic->request_filter2D_roi_end();
						}
						else if (Overlay.getKind() == Autofocus)
						{
							Cd->autofocusZone(texZone, AccessMode::Set);
							Ic->request_autofocus();
							Overlay.setKind(KindOfOverlay::Zoom);
						}
						// TO DO ~ SCRAP FIX ------
						// Noise & Signal are reversed to "fix" PlotWindow graph render
						// An another solution must be found to correct this bug.
						// Normal code :
						/*
						else if (Overlay.getKind() == Signal)
							Cd->signalZone(texZone, AccessMode::Set);
						else if (Overlay.getKind() == Noise)
							Cd->noiseZone(texZone, AccessMode::Set);
						*/
						else if (Overlay.getKind() == Noise)
							Cd->signalZone(texZone, AccessMode::Set);
						else if (Overlay.getKind() == Signal)
							Cd->noiseZone(texZone, AccessMode::Set);
						// ----------------------
						Ic->notify_observers();
					}
				}
			}
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
			static bool initCross = false;

			DirectWindow::keyPressEvent(e);
			if (Cd->stft_view_enabled.load() && e->key() == Qt::Key::Key_Space)
			{
				slicesAreLocked.exchange(!slicesAreLocked.load());
				makeCurrent();
				if (slicesAreLocked.load())
				{
					setCursor(Qt::ArrowCursor);
				}
				else
				{
					setCursor(Qt::CrossCursor);
					if (!initCross)
					{
						Overlay.initCrossBuffer();
						initCross = true;
					}
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
			Overlay.setCrossBuffer(pos, QSize(Fd.width, Fd.height));
		}
	}
}
