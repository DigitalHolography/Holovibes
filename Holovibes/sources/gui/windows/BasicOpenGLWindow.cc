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
#include "HoloWindow.hh"

#include "tools.hh"

namespace holovibes
{
	using camera::FrameDescriptor;
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
			translate_(0.f, 0.f, 0.f, 0.f),
			scale_(1.f),
			angle_(0.f),
			flip_(0),
			cuResource(nullptr),
			cuStream(nullptr),
			cuPtrToPbo(nullptr),
			sizeBuffer(0),
			Program(nullptr),
			Vao(0),
			Vbo(0), Ebo(0), Pbo(0),
			Tex(0),
			overlay_manager_(this),
			transform_matrix_(1.0f),
			transform_inverse_matrix_(1.0f)
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

		const KindOfOverlay BasicOpenGLWindow::getKindOfOverlay() const
		{
			return overlay_manager_.getKind();
		}

		ComputeDescriptor* BasicOpenGLWindow::getCd()
		{
			return Cd;
		}

		const FrameDescriptor& BasicOpenGLWindow::getFd() const
		{
			return Fd;
		}

		OverlayManager& BasicOpenGLWindow::getOverlayManager()
		{
			return overlay_manager_;
		}
		
		const glm::mat3x3 & BasicOpenGLWindow::getTransformMatrix() const
		{
			return transform_matrix_;
		}

		const glm::mat3x3 & BasicOpenGLWindow::getTransformInverseMatrix() const
		{
			return transform_inverse_matrix_;
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
				fullScreen_ = true;
				break;
			case Qt::Key::Key_Escape:
				winPos = QPoint(0, 0);
				winState = Qt::WindowNoState;
				setWindowState(winState);
				fullScreen_ = false;
				break;
			case Qt::Key::Key_8:
				translate_[1] -= 0.1f / scale_;
				setTransform();
				break;
			case Qt::Key::Key_2:
				translate_[1] += 0.1f / scale_;
				setTransform();
				break;
			case Qt::Key::Key_6:
				translate_[0] += 0.1f / scale_;
				setTransform();
				break;
			case Qt::Key::Key_4:
				translate_[0] -= 0.1f / scale_;
				setTransform();
				break;
			}
			overlay_manager_.keyPress(e);
		}

		void	BasicOpenGLWindow::wheelEvent(QWheelEvent *e)
		{
			if (kView != KindOfView::Hologram || !is_between(e->x(), 0, width()) || !is_between(e->y(), 0, height()))
				return;
			const float xGL = (static_cast<float>(e->x() - width() / 2)) / static_cast<float>(width()) * 2.f;
			const float yGL = -((static_cast<float>(e->y() - height() / 2)) / static_cast<float>(height())) * 2.f;
			if (e->angleDelta().y() > 0)
			{
				scale_ += 0.1f * scale_;
				translate_[0] += xGL * 0.1 / scale_;
				translate_[1] += -yGL * 0.1 / scale_;
				setTransform();
			}
			else if (e->angleDelta().y() < 0)
			{
				scale_ -= 0.1f * scale_;
				if (scale_ < 1.f)
					resetTransform();
				else
				{
					translate_[0] -= -xGL * 0.1 / scale_;
					translate_[1] -= yGL * 0.1 / scale_;
					setTransform();
				}
			}
		}

		void	BasicOpenGLWindow::setAngle(float a)
		{
			angle_ = a;
			setTransform();
		}

		float BasicOpenGLWindow::getAngle() const
		{
			return angle_;
		}

		void	BasicOpenGLWindow::setFlip(bool f)
		{
			flip_ = f;
			setTransform();
		}

		bool BasicOpenGLWindow::getFlip() const
		{
			return flip_;
		}

		void BasicOpenGLWindow::setTranslate(float x, float y)
		{
			translate_[0] = x;
			translate_[1] = y;
			setTransform();
		}

		glm::vec2 BasicOpenGLWindow::getTranslate() const
		{
			return glm::vec2(translate_[0], translate_[1]);
		}
		
		void BasicOpenGLWindow::resetTransform()
		{
			translate_ = { 0.f, 0.f, 0.f, 0.f };
			scale_ = 1.f;
			flip_ = false;
			setTransform();
		}

		void BasicOpenGLWindow::setScale(float scale)
		{
			scale_ = scale;
			setTransform();
		}

		float BasicOpenGLWindow::getScale() const
		{
			return scale_;
		}

		void BasicOpenGLWindow::setTransform()
		{
			const glm::mat4 rotY = glm::rotate(glm::mat4(1.f), glm::radians(180.f * (flip_ == 1)), glm::vec3(0.f, 1.f, 0.f));
			const glm::mat4 rotZ = glm::rotate(glm::mat4(1.f), glm::radians(angle_), glm::vec3(0.f, 0.f, 1.f));
			const glm::mat4 rotYZ = rotY * rotZ;

			const glm::mat4 scl = glm::scale(glm::mat4(1.f),
					glm::vec3(kView == KindOfView::SliceYZ ? 1 : scale_,
						kView == KindOfView::SliceXZ ? 1 : scale_,
						1.f));
			glm::mat4 mvp = rotYZ * scl;

			for (uint id = 0; id < 2; id++)
				if (is_between(translate_[id], -FLT_EPSILON, FLT_EPSILON))
					translate_[id] = 0.f;

			glm::vec4 trs = rotYZ * translate_;
			transform_matrix_ = mvp;
			// GLM matrix are column major so the translation vector is in [2][X] and not [X][2]
			transform_matrix_[2][0] = -translate_[0] * 2 * scale_;
			transform_matrix_[2][1] = translate_[1] * 2 * scale_;

			transform_matrix_[2][2] = 1;
			
			transform_inverse_matrix_ = glm::inverse(transform_matrix_);
			if (Program)
			{
				makeCurrent();
				Program->bind();
				Program->setUniformValue(Program->uniformLocation("angle"), angle_);
				Program->setUniformValue(Program->uniformLocation("flip"), flip_);
				Program->setUniformValue(Program->uniformLocation("translate"), trs[0], trs[1]);
				QMatrix4x4 m(glm::value_ptr(mvp));
				Program->setUniformValue(Program->uniformLocation("mvp"), m.transposed());
				Program->release();
			}

			auto holo = dynamic_cast<HoloWindow*>(this);
			if (holo)
				holo->update_slice_transforms();
		}

		void BasicOpenGLWindow::resetSelection()
		{
			overlay_manager_.reset();
		}

		void	BasicOpenGLWindow::setCd(ComputeDescriptor* cd)
		{
			Cd = cd;
		}
	}
}
