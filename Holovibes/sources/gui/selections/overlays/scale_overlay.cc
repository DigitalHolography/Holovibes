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

		void ScaleOverlay::setBuffer()
		{
			auto cd = parent_->getCd();
			auto fd = parent_->getFd();
			const float pix_size = (cd->lambda * cd->zdistance) / (fd.width * cd->pixel_size * 1e-6);

			units::ConversionData convert(parent_);

			// Setting the scale at 5% from bottom and 90% from top
			// Setting the scale at 75% from left and 10% from right
			units::PointOpengl topLeft(convert, 0.5, -0.88);
			units::PointOpengl bottomRight(convert, 0.8, -0.9);

			// Building zone
			zone_ = units::RectFd(topLeft, bottomRight);

			// Retrieving number of pixel contained in the displayed image
			units::PointOpengl topLeft_gl(convert, -1, -1);
			units::PointOpengl bottomRight_gl(convert, 1, -1);
			units::PointFd topLeft_fd = topLeft_gl;
			units::PointFd bottomRight_fd = bottomRight_gl;
			const float nb_pixel = bottomRight_fd.x() - topLeft_fd.x();
			const float size = nb_pixel * pix_size * 0.15f;
			std::cout << "scale: " << size << "m" << std::endl;

			// Updating opengl buffer
			RectOverlay::setBuffer();
		}
	}
}
