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

		void	HoloWindow::wheelEvent(QWheelEvent *e)
		{
			BasicOpenGLWindow::wheelEvent(e);
		}
		
		void	HoloWindow::focusInEvent(QFocusEvent *e)
		{
			QOpenGLWindow::focusInEvent(e);
			Cd->current_window.exchange(WindowKind::XYview);
			Cd->notify_observers();
		}
	}
}
