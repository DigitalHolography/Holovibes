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
			zone_.setBottomRight(QPoint(
				zone_.topLeft().x() +
				min * ((zone_.topLeft().x() < zone_.bottomRight().x()) * 2 - 1),
				zone_.topLeft().y() +
				min * ((zone_.topLeft().y() < zone_.bottomRight().y()) * 2 - 1)
			));
		}

		void Filter2DOverlay::move(QPoint pos, QSize win_size)
		{
			display_ = true;
			zone_.setBottomRight(pos);
			make_square();
			setBuffer(win_size);
		}

		void Filter2DOverlay::checkCorners(ushort frameSide)
		{
			// Resizing the square selection to the window

			if (zone_.bottomRight().x() < 0)
				zone_.setBottomRight(QPoint(0, zone_.bottomRight().y()));
			if (zone_.bottomRight().y() < 0)
				zone_.setBottomRight(QPoint(zone_.bottomRight().x(), 0));

			if (zone_.bottomRight().x() > frameSide)
				zone_.setBottomRight(QPoint(frameSide, zone_.bottomRight().y()));
			if (zone_.bottomRight().y() > frameSide)
				zone_.setBottomRight(QPoint(zone_.bottomRight().x(), frameSide));

			// Making it a square again
			make_square();

			RectOverlay::checkCorners();
		}

		void Filter2DOverlay::release(ushort frameSide)
		{
			checkCorners(parent_->width());

			if (zone_.topLeft() == zone_.bottomRight())
				return;

			Rectangle texZone = getTexZone(frameSide);

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
