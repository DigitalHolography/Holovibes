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
			, elemLineIndex_(0)
			, locked_(true)
			, last_clicked_(units::ConversionData(parent))
			, mouse_position_(units::ConversionData(parent))
			, horizontal_zone_(units::ConversionData(parent))
		{
			color_ = { 1.f, 0.f, 0.f };
			alpha_ = 0.05f;
			display_ = true;
			zoom_ = std::make_shared<ZoomOverlay>(parent_);
			zoom_->initProgram();
		}
		
		CrossOverlay::~CrossOverlay()
		{
			glDeleteBuffers(1, &elemLineIndex_);
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
			if (zoom_ && zoom_->isActive() && zoom_->isDisplayed())
				zoom_->draw();

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

		void CrossOverlay::press(QMouseEvent *e)
		{
			if (zoom_ && zoom_->isActive())
				zoom_->press(e);
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
				units::PointWindow pos_window = getMousePos(e->pos());
				units::PointFd pos = pos_window;
				mouse_position_ = pos;
				std::stringstream ss;
				ss << "(Y,X) = (" << pos.y() << "," << pos.x() << ")";
				InfoManager::get_manager()->update_info("STFT Slice Cursor", ss.str());
				auto cd = parent_->getCd();
				cd->stftCursor(pos, AccessMode::Set);
				// ---------------
				if (cd->x_accu_enabled)
				{
					cd->x_accu_min_level = std::min(pos.x().get(), last_clicked_.x().get());
					cd->x_accu_max_level = std::max(pos.x().get(), last_clicked_.x().get());
				}
				if (cd->y_accu_enabled)
				{
					cd->y_accu_min_level = std::min(pos.y().get(), last_clicked_.y().get());
					cd->y_accu_max_level = std::max(pos.y().get(), last_clicked_.y().get());
				}
				cd->notify_observers();
			}
			else if (zoom_ && zoom_->isActive())
				zoom_->move(e);
		}

		void CrossOverlay::release(ushort frameside)
		{
			if (zoom_ && zoom_->isActive())
				zoom_->release(frameside);
			zoom_ = std::make_shared<ZoomOverlay>(parent_);
			zoom_->initProgram();
		}

		void CrossOverlay::computeZone()
		{
			auto cd = parent_->getCd();
			units::PointFd topLeft;
			units::PointFd bottomRight;
			units::PointFd cursor;
			cd->stftCursor(cursor, Get);

			// Computing min/max coordinates in function of the frame_descriptor
			units::ConversionData convert(parent_);
			units::PointFd min(convert, cd->x_accu_min_level, cd->y_accu_min_level);
			units::PointFd max(convert, cd->x_accu_max_level, cd->y_accu_max_level);

			// Setting the zone_
			if (!cd->x_accu_enabled)
			{
				min.x().set(cursor.x());
				max.x().set(cursor.x());
			}
			if (!cd->y_accu_enabled)
			{
				min.y().set(cursor.y());
				max.y().set(cursor.y());
			}
			max.x() += 1;
			max.y() += 1;
			zone_ = units::RectFd(convert, min.x(), 0, max.x(), parent_->getFd().height);
			horizontal_zone_ = units::RectFd(convert, 0, min.y(), parent_->getFd().width, max.y());
		}

		void CrossOverlay::setBuffer()
		{
			Program_->bind();

			const units::RectOpengl zone_gl = zone_;
			const units::RectOpengl h_zone_gl = horizontal_zone_;

			const float subVertices[] = {
				zone_gl.x(), zone_gl.y(),
				zone_gl.right(), zone_gl.y(),
				zone_gl.right(), zone_gl.bottom(),
				zone_gl.x(), zone_gl.bottom(),

				h_zone_gl.x(), h_zone_gl.y(),
				h_zone_gl.right(), h_zone_gl.y(),
				h_zone_gl.right(), h_zone_gl.bottom(),
				h_zone_gl.x(), h_zone_gl.bottom()
			};

			// Updating the buffer at verticesIndex_ with new coordinates
			glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subVertices), subVertices);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			Program_->release();
		}
	}
}
