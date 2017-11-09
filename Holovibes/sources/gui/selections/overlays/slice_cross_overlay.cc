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
			, last_pIndex_(0, 0)
			, pIndex_(0, 0)
		{
			color_ = { 1.f, 0.f, 0.f };
			alpha_ = 0.05f;
			display_ = true;
		}

		void SliceCrossOverlay::init()
		{
			RectOverlay::init();

			// Set line vertices order

			// Horizontal lines
			std::vector<GLuint> elements {
				0, 1,
				3, 2
			};

			// Vertical lines
			if (parent_->getKindOfView() == SliceYZ)
			{
				elements = {
					0, 3,
					1, 2
				};
			}

			glGenBuffers(1, &elemLineIndex_);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements.data(), GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}

		void SliceCrossOverlay::draw()
		{
			setBuffer();

			Vao_.bind();
			Program_->bind();

			glEnableVertexAttribArray(colorShader_);
			glEnableVertexAttribArray(verticesShader_);

			// Drawing two lines
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
			glUniform1f(glGetUniformLocation(Program_->programId(), "alpha"), line_alpha_);
			glDrawElements(GL_LINES, 4, GL_UNSIGNED_INT, nullptr);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			// Drawing area between two lines
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
			glUniform1f(glGetUniformLocation(Program_->programId(), "alpha"), alpha_);
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
				if (!locked_)
					last_pIndex_ = pIndex_;
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

				// Computing p in function of mouse position
				uint depth = kView == SliceXZ ? parent_->height() : parent_->width();
				pIndex_ = e->pos();
				pIndex_.setX(pIndex_.x() * Cd->nsamples / depth);
				pIndex_.setY(pIndex_.y() * Cd->nsamples / depth);

				uint p = (kView == SliceXZ) ? pIndex_.y() : pIndex_.x();
				uint last_p = (kView == SliceXZ) ? last_pIndex_.y() : last_pIndex_.x();
				Cd->pindex = p;
				if (Cd->p_accu_enabled.load())
				{
					Cd->p_accu_max_level = std::max(p, last_p);
					Cd->p_accu_min_level = std::min(p, last_p);
				}
				Cd->notify_observers();
			}
		}

		void SliceCrossOverlay::release(ushort frameside)
		{

		}

		void SliceCrossOverlay::setBuffer()
		{
			auto cd = parent_->getCd();
			QPoint topLeft;
			QPoint bottomRight;
			auto kView = parent_->getKindOfView();

			// Computing pmin/pmax coordinates in function of the frame_descriptor
			const float side = kView == SliceXZ ? parent_->height() : parent_->width();
			const float ratio = side / cd->nsamples;
			uint pmin = cd->p_accu_min_level;
			uint pmax = cd->p_accu_max_level;

			// Setting the zone_
			if (!cd->p_accu_enabled)
			{
				pmin = cd->pindex;
				pmax = cd->pindex;
			}
			pmin *= ratio;
			pmax = (pmax + 1) * ratio;
			topLeft = (kView == SliceXZ) ? QPoint(0, pmin) : QPoint(pmin, 0);
			bottomRight = (kView == SliceXZ) ? QPoint(parent_->width(), pmax) : QPoint(pmax, parent_->height());
			zone_ = QRect(topLeft, bottomRight);

			// Updating opengl buffer
			RectOverlay::setBuffer();
		}
	}
}
