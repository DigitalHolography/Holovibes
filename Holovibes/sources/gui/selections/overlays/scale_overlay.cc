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

#include "scale_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "real_position.hh"

namespace holovibes
{
	namespace gui
	{
		ScaleOverlay::ScaleOverlay(BasicOpenGLWindow* parent)
			: RectOverlay(KindOfOverlay::Scale, parent)
		{
			color_ = { 1.f, 1.f, 1.f };
			alpha_ = 1.f;
			display_ = true;
		}

		ScaleOverlay::~ScaleOverlay()
		{
		}

        void ScaleOverlay::draw()
        {
			RectOverlay::draw();
			// Do not bind the sader program there. For a strange reason, it hides the text.
			glRasterPos2f(text_position_.x(), text_position_.y());
			glDrawPixels(text_.width(), text_.height(), GL_RGBA, GL_UNSIGNED_BYTE, text_.bits());
		}

		void ScaleOverlay::setBuffer()
		{
			auto cd = parent_->getCd();
			auto fd = parent_->getFd();

			units::ConversionData convert(parent_);

			// Setting the scale from 94% to 95% from top
			// Setting the scale from 80% to 90% from left
			units::PointOpengl topLeft(convert, 0.6f, -0.88f);
			units::PointOpengl bottomRight(convert, 0.8f, -0.9f);

			// Building zone
			scale_zone_ = units::RectOpengl(topLeft, bottomRight);

			units::PointReal real_topLeft = units::PointFd(topLeft);
			units::PointReal real_bottomRight = units::PointFd(bottomRight);

			double size = (real_topLeft - real_bottomRight).distance();

			/* The displaying of the text is done following these steps :
					- Writing the information on a text document.
					- Creating a transparent pixel map
					- Using a painter to paint the text document on the pixel map
					- Converting the pixel map into a QImage and mirroring it (opengl y-axis is reversed)
					- Using glDrawPixels with the Byte buffer of the QImage.
					
				We cannot simply use a QPainter to draw directly the QPixMap because it will alternate between opengl calls
				and qpainter calls. Qt modifies the opengl/openglshaders environment and we didn't manage to fix that.

				Passing via an image to retrieve the bytes buffer and mirror it is a really crap bug fix,
				please fix it when you have a solution */

			const int exponent = std::floor(log10f(size));
			const float significand = size * pow(10, -exponent);
			std::stringstream ss;
			ss << std::fixed << std::setprecision(1) << significand;
			ss << "&#8339;10<sup>" << exponent << "</sup>m";

			QTextDocument td;
			// Text
			QString text(ss.str().c_str());
			td.setHtml(text);

			// Font
			const int base_font_size = 12;
			td.setDefaultFont(QFont("Arial", base_font_size, QFont::ExtraBold));

			// Black outline
			QTextCharFormat format;
			format.setTextOutline(QPen(Qt::black, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
			QTextCursor cursor(&td);
			cursor.select(QTextCursor::Document);
			cursor.mergeCharFormat(format);

			// Pixel map
			QPixmap pixmap = QPixmap(td.size().toSize());
			pixmap.fill(Qt::transparent);
			QPainter pixmap_painter(&pixmap);
			td.drawContents(&pixmap_painter);

			// Position
			units::WindowPixel x_pos = scale_zone_.center().x();
			units::WindowPixel y_pos = topLeft.y();

			// Setting variables
			// Since the image is y-mirrored, we set the bottom left corner.
			auto text_width = td.size().width();
			text_position_ = units::PointWindow(convert, std::min(parent_->width() - text_width,  static_cast<int>(x_pos) - text_width / 2), y_pos);
			
			text_ = pixmap.toImage().mirrored(false, true);

			// Updating opengl buffer
			/* It's a copy and paste of set_buffer of RectOverlay
			The only difference is that the zone is not cast to RectOpengl because scale_overlay is already a RectOpengl */
			parent_->makeCurrent();
			Program_->bind();
			const float subVertices[] = {
				scale_zone_.x(), scale_zone_.y(),
				scale_zone_.right(), scale_zone_.y(),
				scale_zone_.right(), scale_zone_.bottom(),
				scale_zone_.x(), scale_zone_.bottom()
			};

			// Updating the buffer at verticesIndex_ with new coordinates
			glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subVertices), subVertices);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			Program_->release();
		}
	}
}
