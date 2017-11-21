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

#include <iomanip> // setprecision
#include <sstream> // stringstream

#include <qabstracttextdocumentlayout.h>
#include <qtextdocument.h>
#include <QOpenGLPaintDevice>
#include <QPainter>
#include <QPen>

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
			QPainter painter(parent_);
			painter.drawPixmap(pixmap_position_, pixmap_);
			RectOverlay::draw();
        }

		void ScaleOverlay::setBuffer()
		{
			auto cd = parent_->getCd();
			auto fd = parent_->getFd();
			const float pix_size = (cd->lambda * cd->zdistance) / (fd.width * cd->pixel_size * 1e-6);

			units::ConversionData convert(parent_);

			// Setting the scale at 5% from bottom and 94% from top
			// Setting the scale at 75% from left and 10% from right
			units::PointOpengl topLeft(convert, 0.5, -0.88);
			units::PointOpengl bottomRight(convert, 0.8, -0.9);

			// Building zone
			/*zone_ = units::RectFd(topLeft, bottomRight);
			inherited attribute zone_ is not used in scale_overlay because it's a RectFd
			which would caause the scale bar to always be a multiple of a FdPixel
			When we are fully zoomed in, the scale bar will then fill the entire screen width */
			scale_zone_ = units::RectOpengl(topLeft, bottomRight);

			// Retrieving number of pixel contained in the displayed image
			units::PointOpengl topLeft_gl(convert, -1, 1);
			units::PointOpengl bottomRight_gl(convert, 1, -1);
			float left = -1;
			float top = 1;
			convert.transform_to_fd(left, top);
			float right = 1;
			float bottom = -1;
			convert.transform_to_fd(right, bottom);

			const float nb_pixel = sqrt(pow(right - left, 2) + pow(top - bottom, 2)) * fd.width / 2.f;
			const float size = nb_pixel * pix_size * 0.15f; // 0.15f because the scale bar only take 15% of the window width
			const int exponent = std::floor(log10f(size));
			const float significand = size * pow(10, -exponent);
			std::stringstream stream;
			stream << std::fixed << std::setprecision(1) << significand;
			QString significand_str = QString::fromStdString(stream.str());
			QString exponent_str = QString::fromStdString(std::to_string(exponent));

			QTextDocument td;
			QString text = significand_str + " &#8339;10<sup>" + exponent_str + "</sup>m";
			td.setHtml(text);
			const int base_font_size = 10;
			td.setDefaultFont(QFont("Arial", base_font_size));
			const auto base_text_size = td.size();
			const int adjusted_font_size = base_font_size * float(static_cast<units::RectWindow>(scale_zone_).width()) / float(base_text_size.width());
			td.setDefaultFont(QFont("Arial", adjusted_font_size));
			const auto text_size = td.size();
			pixmap_ = QPixmap(text_size.toSize());
			pixmap_.fill(Qt::transparent);
			QPainter pixmap_painter(&pixmap_);

			td.drawContents(&pixmap_painter);
			units::WindowPixel x_pos = scale_zone_.center().x();
			units::WindowPixel y_pos = topLeft.y();

			pixmap_position_ = QPoint(x_pos - td.size().width() / 2, y_pos - td.size().height());

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
