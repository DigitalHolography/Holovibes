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

#include "cross_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		CrossOverlay::CrossOverlay(BasicOpenGLWindow* parent)
			: Overlay(KindOfOverlay::Cross, parent)
			, doubleCross_(false)
		{
			color_ = { 1.f, 0.f, 0.f };
			// corresponding to the line transparency
			alpha_ = 0.5f;
			area_alpha_ = 0.05f;
		}

		void CrossOverlay::setBuffer(QPoint pos, QSize frame)
		{
			Program_->bind();
			const float newX = ((static_cast<float>(pos.x()) - (frame.width() * 0.5f)) / frame.width()) * 2.f;
			const float newY = (-((static_cast<float>(pos.y()) - (frame.height() * 0.5f)) / frame.height())) * 2.f;
			const float vertices[] = {
				newX, 1.f,
				newX, -1.f,
				-1.f, newY,
				1.f, newY,
				newX, 1.f,
				newX, -1.f,
				-1.f, newY,
				1.f, newY,
			};
			glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			Program_->release();
			display_ = true;
			doubleCross_ = false;
		}

		void CrossOverlay::setDoubleBuffer(QPoint pos, QPoint pos2, QSize frame)
		{
			Program_->bind();
			const float newX = ((static_cast<float>(pos.x()) - (frame.width() * 0.5f)) / frame.width()) * 2.f;
			const float newY = (-((static_cast<float>(pos.y()) - (frame.height() * 0.5f)) / frame.height())) * 2.f;
			const float newX2 = ((static_cast<float>(pos2.x()) - (frame.width() * 0.5f)) / frame.width()) * 2.f;
			const float newY2 = (-((static_cast<float>(pos2.y()) - (frame.height() * 0.5f)) / frame.height())) * 2.f;
			const float vertices[] = {
				newX, 1.f,
				newX, -1.f,
				newX2, 1.f,
				newX2, -1.f,
				-1.f, newY,
				1.f, newY,
				-1.f, newY2,
				1.f, newY2,
			};
			glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			Program_->release();
			doubleCross_ = true;
			display_ = true;
		}

		void CrossOverlay::init()
		{
			// Program_ already bound by caller (initProgram)
			
			Vao_.bind();

			// Set vertices position
			const float vertices[] = {
				0.f, 1.f,
				0.f, -1.f,
				-1.f, 0.f,
				1.f, 0.f,
				// Second cross
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f
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
				color_[0], color_[1], color_[2],
				color_[0], color_[1], color_[2],
				color_[0], color_[1], color_[2],
				color_[0], color_[1], color_[2]
			};
			glGenBuffers(1, &colorIndex_);
			glBindBuffer(GL_ARRAY_BUFFER, colorIndex_);
			glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(colorShader_);
			glVertexAttribPointer(colorShader_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
			glDisableVertexAttribArray(colorShader_);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			Vao_.release();

			// Program_ released by caller (initProgram)
		}

		void CrossOverlay::draw()
		{
			switch (parent_->getKindOfView())
			{
			case Hologram:
				drawCross(0, 4);
				break;
			case SliceXZ:
				drawCross(2, 2);
			case SliceYZ:
				drawCross(0, 2);
			default:
				break;
			}
		}
		
		void CrossOverlay::drawCross(GLuint offset, GLsizei count)
		{
			Vao_.bind();
			Program_->bind();
			glEnableVertexAttribArray(verticesShader_);
			glEnableVertexAttribArray(colorShader_);

			glUniform1f(glGetUniformLocation(Program_->programId(), "alpha"), alpha_);

			if (doubleCross_)
			{
				glUniform1f(glGetUniformLocation(Program_->programId(), "alpha"), area_alpha_);

				if (count == 2)
				{
					if (offset == 0)
					{
						glDrawArrays(GL_TRIANGLES, 0, 3);
						glDrawArrays(GL_TRIANGLES, 1, 3);
					}
					else
					{
						glDrawArrays(GL_TRIANGLES, 4, 3);
						glDrawArrays(GL_TRIANGLES, 5, 3);
					}
				}
				else if (count == 4)
				{
					glDrawArrays(GL_TRIANGLES, 0, 3);
					glDrawArrays(GL_TRIANGLES, 1, 3);
					glDrawArrays(GL_TRIANGLES, 4, 3);
					glDrawArrays(GL_TRIANGLES, 5, 3);

					// this should have been coded using glDrawElements,
					// but for some reason it doesn't work...
					// So I had to change the point order to use glDrawArrays

					/*
					int indexes[] = { 0, 1, 4,
					2, 3, 6 };
					glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indexes);
					int indexes2[] = { 2, 3, 6,
					3, 6, 7 };
					glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indexes2);
					*/
				}

				glUniform1f(glGetUniformLocation(Program_->programId(), "alpha"), alpha_);

				if (count == 4)
					glDrawArrays(GL_LINES, 0, 8);
				else
					glDrawArrays(GL_LINES, offset == 0 ? 0 : 4, 4);
			}
			else
				glDrawArrays(GL_LINES, offset, count);
			glDisableVertexAttribArray(colorShader_);
			glDisableVertexAttribArray(verticesShader_);
			Program_->release();
			Vao_.release();
		}
	}
}
