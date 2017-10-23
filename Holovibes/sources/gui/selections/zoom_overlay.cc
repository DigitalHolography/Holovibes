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

#include "zoom_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "DirectWindow.hh"

namespace holovibes
{
	namespace gui
	{
		ZoomOverlay::ZoomOverlay(BasicOpenGLWindow* parent)
			: RectOverlay(KindOfOverlay::Zoom, parent)
		{
			color_ = { 0.f, 0.5f, 0.f };
		}

		void ZoomOverlay::release(ushort frameSide)
		{
			checkCorners();

			if (zone_.topLeft() == zone_.bottomRight())
				return;

			// handle Zoom
			// Since we cannot zoom in slice yet, we have to cast here.
			switch (parent_->getKindOfView())
			{
			case Direct:
			case Hologram:
			{
				DirectWindow* window = dynamic_cast<DirectWindow *>(parent_);
				if (window)
					window->zoomInRect(zone_);
				break;
			}
			// Not implemented yet
			case SliceXZ:
			case SliceYZ:
			default:
				break;
			}
			disable();
		}
	}
}
