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

#include "BasicOpenGLWindow.hh"

namespace gui
{
	BasicOpenGLWindow::BasicOpenGLWindow(QPoint p, QSize s, holovibes::Queue& q, t_KindOfView k) :
		QOpenGLWindow(), QOpenGLFunctions(),
		winPos(p), winSize(s),
		Queue(q),
		kView(k),
		cuResource(nullptr),
		Program(nullptr),
		Vao(0),
		Vbo(0), Ebo(0),
		Tex(0)
	{
		if (cudaStreamCreate(&cuStream) != cudaSuccess)
			cuStream = 0;
		resize(winSize);
		setFramePosition(winPos);
		setIcon(QIcon("icon1.ico"));
		show();
	}

	BasicOpenGLWindow::~BasicOpenGLWindow()
	{
		makeCurrent();

		cudaGraphicsUnregisterResource(cuResource);
		cudaStreamDestroy(cuStream);

		if (Tex) glDeleteBuffers(1, &Tex);
		if (Ebo) glDeleteBuffers(1, &Ebo);
		if (Vbo) glDeleteBuffers(1, &Vbo);
		Vao.destroy();
		delete Program;
	}

	const t_KindOfView	BasicOpenGLWindow::getKindOfView() const
	{
		return kView;
	}

	void BasicOpenGLWindow::keyPressEvent(QKeyEvent* e)
	{
		switch (e->key())
		{
			case Qt::Key::Key_F11:
				setWindowState(Qt::WindowFullScreen);
				break;
			case Qt::Key::Key_Escape:
				setWindowState(Qt::WindowNoState);
				break;
			/*if (kView != KindOfView::Slice)
			{
				case Qt::Key::Key_6:
					;
					break;
			}*/
		}
	}
}