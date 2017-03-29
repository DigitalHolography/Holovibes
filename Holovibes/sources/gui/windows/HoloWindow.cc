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
#include <boost/algorithm/string.hpp>
#include "texture_update.cuh"
#include "HoloWindow.hh"

namespace holovibes
{
	namespace gui
	{
		std::atomic<bool> BasicOpenGLWindow::slicesAreLocked = true;

		HoloWindow::HoloWindow(QPoint p, QSize s, Queue& q, SharedPipe ic, CDescriptor& cd) :
			DirectWindow(p, s, q, KindOfView::Hologram),
			Ic(ic),
			Cd(cd)
		{}

		HoloWindow::~HoloWindow()
		{}

		void	HoloWindow::initShaders()
		{
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/render.vertex.glsl");
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/render.fragment.glsl");
			if (!Program->bind()) std::cerr << "[Error] " << Program->log().toStdString() << '\n';
		}

		void	HoloWindow::mousePressEvent(QMouseEvent* e)
		{
			if (!Cd.stft_view_enabled.load())
				DirectWindow::mousePressEvent(e);
		}

		void	HoloWindow::mouseMoveEvent(QMouseEvent* e)
		{
			if (!Cd.stft_view_enabled.load())
				DirectWindow::mouseMoveEvent(e);
			else if (Cd.stft_view_enabled.load() && !slicesAreLocked.load())
				updateCursorPosition(QPoint(
					e->x() * (Fd.width / static_cast<float>(width())),
					e->y() * (Fd.height / static_cast<float>(height()))));
		}

		void	HoloWindow::mouseReleaseEvent(QMouseEvent* e)
		{
			if (!Cd.stft_view_enabled.load())
			{
				DirectWindow::mouseReleaseEvent(e);
				if (e->button() == Qt::LeftButton)
				{
					if (zoneSelected.getConstZone().topLeft() !=
						zoneSelected.getConstZone().bottomRight())
					{
						if (zoneSelected.getKind() == Filter2D)
						{
							Cd.stftRoiZone(zoneSelected.getTexZone(Fd.width), AccessMode::Set);
							Ic->request_filter2D_roi_update();
							Ic->request_filter2D_roi_end();
						}
						else if (zoneSelected.getKind() == Autofocus)
						{
							Cd.autofocusZone(zoneSelected.getTexZone(Fd.width), AccessMode::Set);
							Ic->request_autofocus();
							zoneSelected.setKind(KindOfSelection::Zoom);
						}
						else if (zoneSelected.getKind() == Signal)
							Cd.signalZone(zoneSelected.getTexZone(Fd.width), AccessMode::Set);
						else if (zoneSelected.getKind() == Noise)
							Cd.noiseZone(zoneSelected.getTexZone(Fd.width), AccessMode::Set);
						Ic->notify_observers();
					}
				}
			}
			else if (e->button() == Qt::RightButton)
				resetTransform();
		}

		void	HoloWindow::keyPressEvent(QKeyEvent* e)
		{
			DirectWindow::keyPressEvent(e);
			if (e->key() == Qt::Key::Key_Space)
			{
				slicesAreLocked.exchange(!slicesAreLocked.load());
				setCursor((slicesAreLocked.load()) ?
					Qt::ArrowCursor : Qt::CrossCursor);
			}
		}

		void	HoloWindow::updateCursorPosition(QPoint pos)
		{
			std::stringstream ss;
			ss << "(Y,X) = (" << pos.y() << "," << pos.x() << ")";
			InfoManager::get_manager()->update_info("STFT Slice Cursor", ss.str());
			Cd.stftCursor(&pos, AccessMode::Set);
		}
	}
}
