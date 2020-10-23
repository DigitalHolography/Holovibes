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

#include "signal_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		SignalOverlay::SignalOverlay(BasicOpenGLWindow* parent)
			: RectOverlay(KindOfOverlay::Signal, parent)
		{
			color_ = { 0.557f, 0.4f, 0.85f };
		}

		void SignalOverlay::release(ushort frameSide)
		{
			if (zone_.height() == 0 || zone_.width() == 0)
				return;

			units::RectFd texZone = zone_;

			// handle Signal
			if (parent_->getKindOfView() == Hologram)
				parent_->getCd()->signalZone(texZone, AccessMode::Set);
		}
	}
}
