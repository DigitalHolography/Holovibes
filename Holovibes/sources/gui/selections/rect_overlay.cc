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

#include "rect_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		RectOverlay::RectOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
			: Overlay(overlay, parent)
		{
		}

		Rectangle RectOverlay::getTexZone(ushort frameSide) const
		{
			return Rectangle(zone_.topLeft() * frameSide / parent_->width(), zone_.size() * frameSide / parent_->width());
		}

		void RectOverlay::init()
		{
			// Program_ already bound by caller (initProgram)

			// Set vertices position
			const float vertices[] = {
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f,
			};
			glGenBuffers(1, &verticesIndex_);
			glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
			glDisableVertexAttribArray(2);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			// Set color
			const float colorData[] = {
				color_[0], color_[1], color_[2],
				color_[0], color_[1], color_[2],
				color_[0], color_[1], color_[2],
				color_[0], color_[1], color_[2],
			};
			glGenBuffers(1, &colorIndex_);
			glBindBuffer(GL_ARRAY_BUFFER, colorIndex_);
			glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(3);
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
			glDisableVertexAttribArray(3);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			// Set vertices order
			const GLuint elements[] = {
				0, 1, 2,
				2, 3, 0,
				4, 5, 6,
				6, 7, 4
			};
			glGenBuffers(1, &elemIndex_);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			// Program_ realsed by caller (initProgram)
		}

		void RectOverlay::draw()
		{
			Program_->bind();
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
			glEnableVertexAttribArray(2);
			glEnableVertexAttribArray(3);

			glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);

			glDisableVertexAttribArray(3);
			glDisableVertexAttribArray(2);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			Program_->release();
		}

		void RectOverlay::checkCorners()
		{
			if (zone_.width() < 0)
			{
				QPoint t0pRight = zone_.topRight();
				QPoint b0ttomLeft = zone_.bottomLeft();

				zone_.setTopLeft(t0pRight);
				zone_.setBottomRight(b0ttomLeft);
			}
			if (zone_.height() < 0)
			{
				QPoint t0pRight = zone_.topRight();
				QPoint b0ttomLeft = zone_.bottomLeft();

				zone_.setTopLeft(b0ttomLeft);
				zone_.setBottomRight(t0pRight);
			}
		}

		void RectOverlay::setBuffer(QSize win_size)
		{
			if (Program_)
			{
				Program_->bind();
				const float w = static_cast<float>(win_size.width());
				const float h = static_cast<float>(win_size.height());
				const float x0 = ((static_cast<float>(zone_.topLeft().x()) - (w * 0.5f)) / w) * 2.f;
				const float y0 = (-((static_cast<float>(zone_.topLeft().y()) - (h * 0.5f)) / h)) * 2.f;
				const float x1 = ((static_cast<float>(zone_.bottomRight().x()) - (w * 0.5f)) / w) * 2.f;
				const float y1 = (-((static_cast<float>(zone_.bottomRight().y()) - (h * 0.5f)) / h)) * 2.f;

				const float subVertices[] = {
					x0, y0,
					x1, y0,
					x1, y1,
					x0, y1
				};
				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subVertices), subVertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program_->release();
			}
		}

		void RectOverlay::move(QPoint pos, QSize win_size)
		{
			zone_.setBottomRight(pos);
			if (display_)
				setBuffer(win_size);
		}
	}
}
