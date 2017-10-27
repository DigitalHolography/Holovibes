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

#include <sstream>
#include "cross_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "info_manager.hh"

namespace holovibes
{
	namespace gui
	{
		CrossOverlay::CrossOverlay(BasicOpenGLWindow* parent)
			: Overlay(KindOfOverlay::Cross, parent)
			, line_alpha_(0.5f)
			, horizontal_zone_(0, 0)
		{
			color_ = { 1.f, 0.f, 0.f };
			alpha_ = 0.05f;
			display_ = true;
		}

		void CrossOverlay::init()
		{
			// Program_ already bound by caller (initProgram)
			
			Vao_.bind();

			// Set vertices position
			const float vertices[] = {
				// vertical area
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f,
				//horizontal area
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
				// vertical area
				color_[0], color_[1], color_[2],
				color_[0], color_[1], color_[2],
				color_[0], color_[1], color_[2],
				color_[0], color_[1], color_[2],
				// horizontal area
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

			// Set line vertices order
			std::vector<GLuint> lineElements{
				// topleft cross
				0, 3,
				4, 5,
				// bottom right cross
				1, 2,
				7, 6
			};
			glGenBuffers(1, &elemLineIndex_);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, lineElements.size() * sizeof(GLuint), lineElements.data(), GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			// Set rectangle vertices order
			std::vector<GLuint> elements{
				// vertical area
				0, 1, 2,
				2, 3, 0,
				//horizontal area
				4, 5, 6,
				6, 7, 4
			};
			glGenBuffers(1, &elemIndex_);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements.size() * sizeof(GLuint), elements.data(), GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			Vao_.release();

			// Program_ released by caller (initProgram)
		}

		void CrossOverlay::draw()
		{
			computeZone();
			setBuffer();

			Vao_.bind();
			Program_->bind();

			glEnableVertexAttribArray(colorShader_);
			glEnableVertexAttribArray(verticesShader_);

			// Drawing four lines
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
			glUniform1f(glGetUniformLocation(Program_->programId(), "alpha"), line_alpha_);
			glDrawElements(GL_LINES, 8, GL_UNSIGNED_INT, nullptr);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			// Drawing areas between lines
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
			glUniform1f(glGetUniformLocation(Program_->programId(), "alpha"), alpha_);
			glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, nullptr);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			glDisableVertexAttribArray(verticesShader_);
			glDisableVertexAttribArray(colorShader_);

			Program_->release();
			Vao_.release();
		}

		void CrossOverlay::keyPress(QKeyEvent *e)
		{
			if (e->key() == Qt::Key_Space)
			{
				if (!locked_)
					last_clicked_ = mouse_position_;
				locked_ = !locked_;
				parent_->setCursor(locked_ ? Qt::ArrowCursor : Qt::CrossCursor);
			}
		}

		void CrossOverlay::move(QMouseEvent *e)
		{
			if (!locked_)
			{
				auto fd = parent_->getFd();
				auto pos = getMousePos(e->pos());
				pos.setX(pos.x() * fd.width / parent_->width());
				pos.setY(pos.y() * fd.height / parent_->height());
				mouse_position_ = pos;
				std::stringstream ss;
				ss << "(Y,X) = (" << pos.y() << "," << pos.x() << ")";
				InfoManager::get_manager()->update_info("STFT Slice Cursor", ss.str());
				auto cd = parent_->getCd();
				cd->stftCursor(&pos, AccessMode::Set);
				// ---------------
				if (cd->x_accu_enabled)
				{
					cd->x_accu_min_level = std::min(pos.x(), last_clicked_.x());
					cd->x_accu_max_level = std::max(pos.x(), last_clicked_.x());
				}
				if (cd->y_accu_enabled)
				{
					cd->y_accu_min_level = std::min(pos.y(), last_clicked_.y());
					cd->y_accu_max_level = std::max(pos.y(), last_clicked_.y());
				}
				cd->notify_observers();
			}
		}

		void CrossOverlay::release(ushort frameside)
		{

		}

		void CrossOverlay::computeZone()
		{
			auto cd = parent_->getCd();
			QPoint topLeft;
			QPoint bottomRight;
			QPoint cursor;
			cd->stftCursor(&cursor, Get);

			// Computing min/max coordinates in function of the frame_descriptor
			auto frame_desc = parent_->getFd();
			const float ratioX = (float)(parent_->width()) / (frame_desc.width - 1);
			const float ratioY = (float)(parent_->height()) / (frame_desc.height - 1);
			uint xmin = cd->x_accu_min_level;
			uint xmax = cd->x_accu_max_level;
			uint ymin = cd->y_accu_min_level;
			uint ymax = cd->y_accu_max_level;

			// Setting the zone_
			if (!cd->x_accu_enabled)
			{
				xmin = cursor.x();
				xmax = cursor.x();
			}
			if (!cd->y_accu_enabled)
			{
				ymin = cursor.y();
				ymax = cursor.y();
			}
			xmin *= ratioX;
			xmax = (xmax + 1) * ratioX;
			ymin *= ratioY;
			ymax = (ymax + 1) * ratioY;
			zone_ = QRect(QPoint(xmin, 0), QPoint(xmax, parent_->height()));
			horizontal_zone_ = QRect(QPoint(0, ymin), QPoint(parent_->width(), ymax));
		}

		void CrossOverlay::setBuffer()
		{
			Program_->bind();
			QSize win_size = parent_->size();
			const float w = win_size.width();
			const float h = win_size.height();

			// Normalizing the zones to (-1; 1)
			const float x0 = 2.f * zone_.topLeft().x() / w - 1.f;
			const float y0 = -(2.f * zone_.topLeft().y() / h - 1.f);
			const float x1 = 2.f * zone_.bottomRight().x() / w - 1.f;
			const float y1 = -(2.f * zone_.bottomRight().y() / h - 1.f);

			const float x2 = 2.f * horizontal_zone_.topLeft().x() / w - 1.f;
			const float y2 = -(2.f * horizontal_zone_.topLeft().y() / h - 1.f);
			const float x3 = 2.f * horizontal_zone_.bottomRight().x() / w - 1.f;
			const float y3 = -(2.f * horizontal_zone_.bottomRight().y() / h - 1.f);

			const float subVertices[] = {
				x0, y0,
				x1, y0,
				x1, y1,
				x0, y1,

				x2, y2,
				x3, y2,
				x3, y3,
				x2, y3
			};

			// Updating the buffer at verticesIndex_ with new coordinates
			glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subVertices), subVertices);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			Program_->release();
		}
	}
}
