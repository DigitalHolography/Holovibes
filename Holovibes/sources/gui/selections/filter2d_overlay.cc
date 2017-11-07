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

#include "filter2d_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "HoloWindow.hh"

namespace holovibes
{
	namespace gui
	{
		Filter2DOverlay::Filter2DOverlay(BasicOpenGLWindow* parent)
			: RectOverlay(KindOfOverlay::Filter2D, parent)
		{
			color_ = { 0.f, 0.62f, 1.f };
		}

		void Filter2DOverlay::make_square()
		{
			// Set the bottomRight corner to have a square selection.
			// Since topLeft is always the origin, bottomRight correspond to destination,
			// and can be in every corner (bottomRight can be in the top left corner).
			const int min = std::min(std::abs(zone_.width()), std::abs(zone_.height()));
			zone_.setBottomRight(units::PointWindow(units::ConversionData(parent_),
				zone_.topLeft().x() +
				min * ((zone_.topLeft().x() < zone_.bottomRight().x()) * 2 - 1),
				zone_.topLeft().y() +
				min * ((zone_.topLeft().y() < zone_.bottomRight().y()) * 2 - 1)
			));
		}

		void Filter2DOverlay::move(QMouseEvent *e)
		{
			if (e->buttons() == Qt::LeftButton)
			{
				display_ = true;
				zone_.setBottomRight(getMousePos(e->pos()));
				make_square();
				setBuffer();
			}
		}

		void Filter2DOverlay::checkCorners(ushort frameSide)
		{
			// Resizing the square selection to the window

			if (zone_.right() < 0)
				zone_.setRight(0);
			if (zone_.bottom() < 0)
				zone_.setBottom(0);

			if (zone_.right() > frameSide)
				zone_.setRight(frameSide);
			if (zone_.bottom() > frameSide)
				zone_.setBottom(frameSide);

			// Making it a square again
			make_square();

			RectOverlay::checkCorners();
		}

		void Filter2DOverlay::release(ushort frameSide)
		{
			checkCorners(parent_->width());

			if (zone_.topLeft() == zone_.bottomRight())
				return;

			units::RectFd texZone = zone_;

			// handle Filter2D
			if (parent_->getKindOfView() == Hologram)
			{
				auto window = dynamic_cast<HoloWindow *>(parent_);
				if (window)
				{
					window->getCd()->stftRoiZone(texZone, AccessMode::Set);
					window->getPipe()->request_filter2D_roi_update();
					window->getPipe()->request_filter2D_roi_end();
				}
			}

			active_ = false;
		}
	}
}
