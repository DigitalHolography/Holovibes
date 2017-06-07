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

#include <QApplication.h>
#include <qdesktopwidget.h>
#include "texture_update.cuh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		BasicOpenGLWindow::BasicOpenGLWindow(QPoint p, QSize s, Queue& q, KindOfView k) :
			QOpenGLWindow(), QOpenGLFunctions(),
			State(Qt::WindowNoState),
			Qu(q),
			Fd(Qu.get_frame_desc()),
			kView(k),
			Translate{ 0.f, 0.f },
			Scale(1.f),
			cuResource(nullptr),
			cuStream(nullptr),
			cuPtrToPbo(nullptr),
			sizeBuffer(0),
			Program(nullptr),
			Vao(0),
			Vbo(0), Ebo(0), Pbo(0),
			Tex(0),
			Overlay()
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

		const KindOfView	BasicOpenGLWindow::getKindOfView() const
		{
			return kView;
		}

		void	BasicOpenGLWindow::setKindOfOverlay(KindOfOverlay k)
		{
			Overlay.setKind(k);
		}

		const KindOfOverlay	BasicOpenGLWindow::getKindOfOverlay() const
		{
			return Overlay.getKind();
		}
		
		void	BasicOpenGLWindow::resizeGL(int width, int height)
		{
			glViewport(0, 0, width, height);
		}

		void	BasicOpenGLWindow::timerEvent(QTimerEvent *e)
		{
			QPaintDeviceWindow::update();
		}

		void	BasicOpenGLWindow::keyPressEvent(QKeyEvent* e)
		{
			static const QRect screen = QApplication::desktop()->availableGeometry();
			switch (e->key())
			{
			case Qt::Key::Key_F11:
				State = Qt::WindowFullScreen;
				setWindowState(State);
				updateDisplaySquare();
				break;
			case Qt::Key::Key_Escape:
				State = Qt::WindowNoState;
				setWindowState(State);
				updateDisplaySquare();
				break;
			case Qt::Key::Key_8:
				Translate[1] -= 0.1f / Scale;
				setTranslate();
				break;
			case Qt::Key::Key_2:
				Translate[1] += 0.1f / Scale;
				setTranslate();
				break;
			case Qt::Key::Key_6:
				Translate[0] += 0.1f / Scale;
				setTranslate();
				break;
			case Qt::Key::Key_4:
				Translate[0] -= 0.1f / Scale;
				setTranslate();
				break;
			}
		}

		void	BasicOpenGLWindow::updateDisplaySquare()
		{
			if (Program && Vbo)
			{
				const QRect			screen = QApplication::desktop()->availableGeometry();
				std::vector<float>	vec;

				makeCurrent();
				Program->bind();

				if (State == Qt::WindowFullScreen)
				{
					const float x_ = 1.f - (static_cast<float>((screen.width() / 2 - screen.height() / 2)) /
						static_cast<float>((screen.width() / 2)));
					std::cout << x_ << std::endl;
					vec = {
						-x_, 1.f,
						0.0f, 0.0f,
						x_, 1.f,
						1.f, 0.0f,
						x_, -1.f,
						1.f, 1.f,
						-x_, -1.f,
						0.0f, 1.f };
				}
				else
				{
					vec = {
						-1.f, 1.f,
						0.0f, 0.0f,
						1.f, 1.f,
						1.f, 0.0f,
						1.f, -1.f,
						1.f, 1.f,
						-1.f, -1.f,
						0.0f, 1.f };
				}
				glBindBuffer(GL_ARRAY_BUFFER, Vbo);
				glBufferSubData(GL_ARRAY_BUFFER, 0, 16 * sizeof(float), vec.data());
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program->release();
			}
		}

		void	BasicOpenGLWindow::wheelEvent(QWheelEvent *e)
		{
			if (e->x() < width() && e->y() < height())
			{
				const float xGL = (static_cast<float>(e->x() - width() / 2)) / static_cast<float>(width()) * 2.f;
				const float yGL = -((static_cast<float>(e->y() - height() / 2)) / static_cast<float>(height())) * 2.f;
				if (e->angleDelta().y() > 0)
				{
					Scale += 0.1f * Scale;
					setScale();
					Translate[0] += xGL * 0.1 / Scale;
					Translate[1] += -yGL * 0.1 / Scale;
					setTranslate();
				}
				else if (e->angleDelta().y() < 0)
				{
					Scale -= 0.1f * Scale;
					if (Scale < 1.f)
						resetTransform();
					else
					{
						setScale();
						Translate[0] -= -xGL * 0.1 / Scale;
						Translate[1] -= yGL * 0.1 / Scale;
						setTranslate();
					}
				}
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

		void	BasicOpenGLWindow::setAngle(float a)
		{
			Angle = a;
			if (Program)
			{
				makeCurrent();
				Program->bind();
				glUniform1f(glGetUniformLocation(Program->programId(), "angle"), Angle * (M_PI / 180.f));
				Program->release();
			}
		}

		void	BasicOpenGLWindow::setFlip(int f)
		{
			Flip = f;
			if (Program)
			{
				makeCurrent();
				Program->bind();
				glUniform1i(glGetUniformLocation(Program->programId(), "flip"), Flip);
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

		void	BasicOpenGLWindow::resetSelection()
		{
			makeCurrent();
			Overlay.resetVerticesBuffer();
		}
	}
}
