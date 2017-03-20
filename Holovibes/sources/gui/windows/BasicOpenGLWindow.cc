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

#include "texture_update.cuh"
#include "BasicOpenGLWindow.hh"

namespace gui
{
	BasicOpenGLWindow::BasicOpenGLWindow(QPoint p, QSize s, holovibes::Queue& q, KindOfView k) :
		/* ~~~~~~~~~~~~ */
		QOpenGLWindow(), QOpenGLFunctions(),
		Queue(q),
		Fd(Queue.get_frame_desc()),
		kView(k),
		Translate{ 0.f, 0.f },
		Scale(1.f),
		cuResource(nullptr),
		cuStream(nullptr),
		cuArray(nullptr),
		cuSurface(0),
		cuPtrToPbo(nullptr),
		sizeBuffer(0),
		Program(nullptr),
		Vao(0),
		Vbo(0), Ebo(0), Pbo(0),
		Tex(0),
		zoneSelected()
	{
		if (cudaStreamCreate(&cuStream) != cudaSuccess)
			cuStream = nullptr;
		resize(s);
		setFramePosition(p);
		setIcon(QIcon("icon1.ico"));
		show();
	}

	BasicOpenGLWindow::~BasicOpenGLWindow()
	{
		makeCurrent();

		cudaGraphicsUnregisterResource(cuResource);
		cudaStreamDestroy(cuStream);

		if (Tex) glDeleteBuffers(1, &Tex);
		if (Pbo) glDeleteBuffers(1, &Pbo);
		if (Ebo) glDeleteBuffers(1, &Ebo);
		if (Vbo) glDeleteBuffers(1, &Vbo);
		Vao.destroy();
		delete Program;
	}

	const	KindOfView	BasicOpenGLWindow::getKindOfView() const
	{
		return kView;
	}

	void	BasicOpenGLWindow::setKindOfSelection(KindOfSelection k)
	{
		zoneSelected.setKind(k);
	}

	const KindOfSelection BasicOpenGLWindow::getKindOfSelection() const
	{
		return zoneSelected.getKind();
	}

	void	BasicOpenGLWindow::timerEvent(QTimerEvent *e)
	{
		QPaintDeviceWindow::update();
	}

	void	BasicOpenGLWindow::keyPressEvent(QKeyEvent* e)
	{
		switch (e->key())
		{
			case Qt::Key::Key_F11:
				setWindowState(Qt::WindowFullScreen);
				break;
			case Qt::Key::Key_Escape:
				setWindowState(Qt::WindowNoState);
				break;
			case (Qt::Key::Key_Space):
				if (kView == Hologram)
				{
					slicesAreLocked.exchange(!slicesAreLocked.load());
					setCursor((slicesAreLocked.load()) ?
						Qt::ArrowCursor : Qt::CrossCursor);
				}
				break;
			case Qt::Key::Key_Up :
				Translate[1] -= 0.1f / Scale;
				setTranslate();
				break;
			case Qt::Key::Key_Down :
				Translate[1] += 0.1f / Scale;
				setTranslate();
				break;
			case Qt::Key::Key_Right:
				Translate[0] += 0.1f / Scale;
				setTranslate();
				break;
			case Qt::Key::Key_Left:
				Translate[0] -= 0.1f / Scale;
				setTranslate();
				break;
		}
	}

	void	BasicOpenGLWindow::setTranslate()
	{
		for (uint id = 0; id < 2; id++)
			Translate[id] = ((Translate[id] > 0 && Translate[id] < FLT_EPSILON) ||
							(Translate[id] < 0 && Translate[id] > -FLT_EPSILON)) ?
								0.f : Translate[id];
		if (Program)
		{
			makeCurrent();
			Program->bind();
			glUniform2f(glGetUniformLocation(Program->programId(), "translate"), Translate[0], Translate[1]);
			Program->release();
		}
	}
	
	void	BasicOpenGLWindow::setScale()
	{
		if (Program)
		{
			makeCurrent();
			Program->bind();
			glUniform1f(glGetUniformLocation(Program->programId(), "scale"), Scale);
			Program->release();
		}
	}

	void	BasicOpenGLWindow::resetTransform()
	{
		Translate = { 0.f, 0.f };
		Scale = 1.f;
		if (Program)
		{
			makeCurrent();
			Program->bind();
			glUniform1f(glGetUniformLocation(Program->programId(), "scale"), Scale);
			glUniform2f(glGetUniformLocation(Program->programId(), "translate"), Translate[0], Translate[1]);
			Program->release();
		}
	}
}