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

#include "square_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		SquareOverlay::SquareOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
			: RectOverlay(overlay, parent)
		{
		}

		void SquareOverlay::make_square()
		{
			// Set the bottomRight corner to have a square selection.
			const int min = std::min(std::abs(zone_.width()), std::abs(zone_.height()));
			zone_.setDst(units::PointFd(units::ConversionData(parent_),
				zone_.src().x() + ((zone_.src().x() < zone_.dst().x()) ? min : -min),
				zone_.src().y() + ((zone_.src().y() < zone_.dst().y()) ? min : -min)
			));
		}

		void SquareOverlay::checkCorners()
		{
			ushort frameSide = parent_->getFd().width;

			// Resizing the square selection to the window
			if (zone_.dst().x() < 0)
				zone_.dstRef().x().set(0);
			else if (zone_.dst().x() > frameSide)
				zone_.dstRef().x().set(frameSide);
			
			if (zone_.dst().y() < 0)
				zone_.dstRef().y().set(0);
			else if (zone_.dst().y() > frameSide)
				zone_.dstRef().y().set(frameSide);

			// Making it a square again
			make_square();
		}

		void SquareOverlay::move(QMouseEvent* e)
		{
			if (e->buttons() == Qt::LeftButton)
			{
				auto pos = getMousePos(e->pos());
				zone_.setDst(pos);
				make_square();
				setBuffer();
				display_ = true;
			}
		}
	}
}
