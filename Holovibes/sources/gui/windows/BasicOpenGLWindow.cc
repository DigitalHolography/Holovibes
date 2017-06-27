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

#include <glm\gtc\type_ptr.hpp>

#include "texture_update.cuh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		BasicOpenGLWindow::BasicOpenGLWindow(QPoint p, QSize s, Queue& q, KindOfView k) :
			QOpenGLWindow(), QOpenGLFunctions(),
			winState(Qt::WindowNoState),
			winPos(p),
			Qu(q),
			Cd(nullptr),
			Fd(Qu.get_frame_desc()),
			kView(k),
			Translate(0.f, 0.f, 0.f, 0.f),
			Scale(1.f),
			Angle(0.f),
			Flip(0),
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
			setIcon(QIcon("Holovibes.ico"));
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
			const QRect screen = QApplication::desktop()->availableGeometry();
			switch (e->key())
			{
			case Qt::Key::Key_F11:
				winPos = QPoint(((screen.width() / 2 - screen.height() / 2)), 0);
				winState = Qt::WindowFullScreen;
				setWindowState(winState);
				break;
			case Qt::Key::Key_Escape:
				winPos = QPoint(0, 0);
				winState = Qt::WindowNoState;
				setWindowState(winState);
				break;
			case Qt::Key::Key_8:
				Translate[1] -= 0.1f / Scale;
				setTransform();
				break;
			case Qt::Key::Key_2:
				Translate[1] += 0.1f / Scale;
				setTransform();
				break;
			case Qt::Key::Key_6:
				Translate[0] += 0.1f / Scale;
				setTransform();
				break;
			case Qt::Key::Key_4:
				Translate[0] -= 0.1f / Scale;
				setTransform();
				break;
			}
		}

		void	BasicOpenGLWindow::wheelEvent(QWheelEvent *e)
		{
			if (kView != KindOfView::SliceXZ && kView != KindOfView::SliceYZ &&
				e->x() < width() && e->y() < height())
			{
				const float xGL = (static_cast<float>(e->x() - width() / 2)) / static_cast<float>(width()) * 2.f;
				const float yGL = -((static_cast<float>(e->y() - height() / 2)) / static_cast<float>(height())) * 2.f;
				if (e->angleDelta().y() > 0)
				{
					Scale += 0.1f * Scale;
					Translate[0] += xGL * 0.1 / Scale;
					Translate[1] += -yGL * 0.1 / Scale;
					setTransform();
				}
				else if (e->angleDelta().y() < 0)
				{
					Scale -= 0.1f * Scale;
					if (Scale < 1.f)
						resetTransform();
					else
					{
						Translate[0] -= -xGL * 0.1 / Scale;
						Translate[1] -= yGL * 0.1 / Scale;
						setTransform();
					}
				}
			}
		}
				
		void	BasicOpenGLWindow::setAngle(float a)
		{
			Angle = a;
			setTransform();
		}

		void	BasicOpenGLWindow::setFlip(int f)
		{
			Flip = f;
			setTransform();
		}

		void	BasicOpenGLWindow::setTransform()
		{
			const glm::mat4 rotY = glm::rotate(glm::mat4(1.f), glm::radians(180.f * (Flip == 1)), glm::vec3(0.f, 1.f, 0.f));
			const glm::mat4 rotZ = glm::rotate(glm::mat4(1.f), glm::radians(Angle), glm::vec3(0.f, 0.f, 1.f));
			const glm::mat4 rotYZ = rotY * rotZ;

			const glm::mat4 scl = glm::scale(glm::mat4(1.f), glm::vec3(Scale, Scale, 1.f));
			glm::mat4 mvp = rotYZ * scl;

			for (uint id = 0; id < 2; id++)
				Translate[id] = ((Translate[id] > 0 && Translate[id] < FLT_EPSILON) ||
				(Translate[id] < 0 && Translate[id] > -FLT_EPSILON)) ?
				0.f : Translate[id];

			glm::vec4 trs = rotYZ * Translate;

			if (Program)
			{
				makeCurrent();
				Program->bind();
				glUniform1f(glGetUniformLocation(Program->programId(), "angle"), Angle);
				glUniform1i(glGetUniformLocation(Program->programId(), "flip"), Flip);
				glUniform2f(
					glGetUniformLocation(Program->programId(), "translate"),
					trs[0], trs[1]);
				glUniformMatrix4fv(
					glGetUniformLocation(Program->programId(), "mvp"),
					1, GL_FALSE,
					glm::value_ptr(mvp));
				Program->release();
			}
		}

		void	BasicOpenGLWindow::resetTransform()
		{
			Translate = { 0.f, 0.f, 0.f, 0.f };
			Scale = 1.f;
			setTransform();
		}

		void	BasicOpenGLWindow::resetSelection()
		{
			makeCurrent();
			Overlay.resetVerticesBuffer();
		}

		void	BasicOpenGLWindow::setCd(ComputeDescriptor* cd)
		{
			Cd = cd;
		}
	}
}
