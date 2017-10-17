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

#include "autofocus_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		AutofocusOverlay::AutofocusOverlay()
			: RectOverlay(KindOfOverlay::Autofocus)
		{
			color_ = { 1.f, 0.6f, 1.f };
		}

		void AutofocusOverlay::release(ushort frameSide)
		{
			checkCorners();

			if (zone_.topLeft() == zone_.bottomRight())
				return;

			Rectangle texZone = getTexZone(frameSide);

			// handle Autofocus
			if (parent_->getKindOfView() == Hologram)
			{
				auto window = dynamic_cast<HoloWindow *>(parent_.get());
				if (window)
				{
					parent_->getCd->autofocusZone(texZone, AccessMode::Set);
					window->getPipe->request_autofocus();
				}
			}
		}
	}
}
