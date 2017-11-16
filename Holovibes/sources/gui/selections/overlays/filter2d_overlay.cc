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
			: SquareOverlay(KindOfOverlay::Filter2D, parent)
		{
			color_ = { 0.f, 0.62f, 1.f };
		}

		void Filter2DOverlay::release(ushort frameSide)
		{
			checkCorners();

			if (zone_.src() == zone_.dst())
				return;

			// handle Filter2D
			auto window = dynamic_cast<HoloWindow *>(parent_);
			if (window)
			{
				window->getCd()->stftRoiZone(zone_, AccessMode::Set);
				window->getPipe()->request_filter2D_roi_update();
				window->getPipe()->request_filter2D_roi_end();
			}

			active_ = false;
		}
	}
}
