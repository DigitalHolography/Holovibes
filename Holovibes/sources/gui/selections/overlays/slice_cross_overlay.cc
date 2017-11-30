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

#include "slice_cross_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		SliceCrossOverlay::SliceCrossOverlay(BasicOpenGLWindow* parent)
			: RectOverlay(KindOfOverlay::SliceCross, parent)
			, line_alpha_(0.5f)
			, elemLineIndex_(0)
			, locked_(true)
			, pIndex_(0, 0)
		{
			color_ = { 1.f, 0.f, 0.f };
			alpha_ = 0.05f;
			display_ = true;
		}

		SliceCrossOverlay::~SliceCrossOverlay()
		{
			parent_->makeCurrent();
			glDeleteBuffers(1, &elemLineIndex_);
		}

		void SliceCrossOverlay::init()
		{
			RectOverlay::init();

			// Set line vertices order
			const GLuint elements[] = {
				0, 1,
				1, 2,
				2, 3,
				3, 0
			};

			glGenBuffers(1, &elemLineIndex_);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}

		void SliceCrossOverlay::draw()
		{
			parent_->makeCurrent();
			setBuffer();
			Vao_.bind();
			Program_->bind();

			glEnableVertexAttribArray(colorShader_);
			glEnableVertexAttribArray(verticesShader_);

			// Drawing two lines
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
			Program_->setUniformValue(Program_->uniformLocation("alpha"), line_alpha_);
			glDrawElements(GL_LINES, 8, GL_UNSIGNED_INT, nullptr);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			// Drawing area between two lines
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
			Program_->setUniformValue(Program_->uniformLocation("alpha"), alpha_);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			glDisableVertexAttribArray(verticesShader_);
			glDisableVertexAttribArray(colorShader_);

			Program_->release();
			Vao_.release();
		}

		void SliceCrossOverlay::keyPress(QKeyEvent *e)
		{
			if (e->key() == Qt::Key_Space)
			{
				locked_ = !locked_;
				parent_->setCursor(locked_ ? Qt::ArrowCursor : Qt::CrossCursor);
			}
		}

		void SliceCrossOverlay::move(QMouseEvent *e)
		{
			if (!locked_)
			{
				auto kView = parent_->getKindOfView();
				auto Cd = parent_->getCd();

				pIndex_ = getMousePos(e->pos());

				uint p = (kView == SliceXZ) ? pIndex_.y() : pIndex_.x();
				Cd->pindex = p;
				Cd->notify_observers();
			}
		}

		void SliceCrossOverlay::release(ushort frameside)
		{

		}

		void SliceCrossOverlay::setBuffer()
		{
			auto cd = parent_->getCd();
			units::PointFd topLeft;
			units::PointFd bottomRight;
			auto kView = parent_->getKindOfView();

			uint pmin = cd->pindex;
			uint pmax = pmin;
			if (cd->p_accu_enabled)
				pmax += cd->p_acc_level;

			units::ConversionData convert(parent_);

			pmax = (pmax + 1);
			topLeft = (kView == SliceXZ) ? units::PointFd(convert, 0, pmin) : units::PointFd(convert, pmin, 0);
			bottomRight = (kView == SliceXZ) ? units::PointFd(convert, parent_->getFd().width, pmax) : units::PointFd(convert, pmax, parent_->getFd().height);
			zone_ = units::RectFd(topLeft, bottomRight);

			// Updating opengl buffer
			RectOverlay::setBuffer();
		}
	}
}
