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
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		RectOverlay::RectOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
			: Overlay(overlay, parent)
		{
		}

		void RectOverlay::init()
		{
			// Program_ already bound by caller (initProgram)

			Vao_.bind();

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
			glEnableVertexAttribArray(verticesShader_);
			glVertexAttribPointer(verticesShader_, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
			glDisableVertexAttribArray(verticesShader_);
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
			glEnableVertexAttribArray(colorShader_);
			glVertexAttribPointer(colorShader_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
			glDisableVertexAttribArray(colorShader_);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			// Set vertices order
			const GLuint elements[] = {
				0, 1, 2,
				2, 3, 0
			};
			glGenBuffers(1, &elemIndex_);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			Vao_.release();

			// Program_ released by caller (initProgram)
		}

		void RectOverlay::draw()
		{
			Vao_.bind();
			Program_->bind();
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
			glEnableVertexAttribArray(colorShader_);
			glEnableVertexAttribArray(verticesShader_);
			glUniform1f(glGetUniformLocation(Program_->programId(), "alpha"), alpha_);

			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

			glDisableVertexAttribArray(verticesShader_);
			glDisableVertexAttribArray(colorShader_);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			Program_->release();
			Vao_.release();
		}

		void RectOverlay::checkCorners()
		{
			if (zone_.width() < 0)
			{
				auto topRight = zone_.topRight();
				auto bottomLeft = zone_.bottomLeft();

				zone_.setTopLeft(topRight);
				zone_.setBottomRight(bottomLeft);
			}
			if (zone_.height() < 0)
			{
				auto topRight = zone_.topRight();
				auto bottomLeft = zone_.bottomLeft();

				zone_.setTopLeft(bottomLeft);
				zone_.setBottomRight(topRight);
			}
		}

		void RectOverlay::setBuffer()
		{
			Program_->bind();

			// Normalizing the zone to (-1; 1)
			units::RectOpengl zone_gl = zone_;

			const float subVertices[] = {
				zone_gl.x(), zone_gl.y(),
				zone_gl.right(), zone_gl.y(),
				zone_gl.right(), zone_gl.bottom(),
				zone_gl.x(), zone_gl.bottom()
			};

			// Updating the buffer at verticesIndex_ with new coordinates
			glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subVertices), subVertices);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			Program_->release();
		}

		void RectOverlay::move(QMouseEvent* e)
		{
			if (e->buttons() == Qt::LeftButton)
			{
				auto pos = getMousePos(e->pos());
				zone_.setBottomRight(pos);
				setBuffer();
				display_ = true;
			}
		}

		void RectOverlay::setZone(units::RectWindow rect, ushort frameside)
		{
			zone_ = rect;
			setBuffer();
			display_ = true;
			release(frameside);
		}
	}
}
