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

#include "filter2d_subzone_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "HoloWindow.hh"

namespace holovibes
{
	namespace gui
	{
		Filter2DSubZoneOverlay::Filter2DSubZoneOverlay(BasicOpenGLWindow* parent)
			: RectOverlay(KindOfOverlay::Filter2DSubZone, parent)
		{
			color_ = { 0.62f, 0.f, 1.f };
		}

		void Filter2DSubZoneOverlay::release(ushort frameSide)
		{
			checkCorners();

			if (zone_.src() == zone_.dst())
				return;

			// handle Filter2D
			auto window = dynamic_cast<HoloWindow *>(parent_);
			if (window)
			{
				window->getCd()->setFilter2DSubZone(zone_);
				window->getPipe()->request_filter2D_roi_update();
				window->getPipe()->request_filter2D_roi_end();
			}

			parent_->getCd()->fft_shift_enabled = false;

            filter2d_overlay_->disable();

			active_ = false;
		}

        void Filter2DSubZoneOverlay::setFilter2dOverlay(std::shared_ptr<Filter2DOverlay> rhs)
        {
            filter2d_overlay_ = rhs;
        }

	}
}
