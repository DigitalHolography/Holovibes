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

#pragma once

#include "compute_descriptor.hh"
#include "Overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class OverlayManager
		{
		public:
			OverlayManager(WindowKind view);
			~OverlayManager();


			void create_autofocus();
			void create_zoom();
			void create_filter2D();
			void create_noise();
			void create_signal();
			void create_cross();

			void disable_all(KindOfOverlay ko);

			void press(QPoint pos);
			void move(QPoint pos, QSize size);
			void release(ushort frameSide);

		private:
			void create_overlay(std::shared_ptr<Overlay> new_overlay);

			std::vector<std::shared_ptr<Overlay>> overlays_;
			WindowKind view_;
			std::shared_ptr<Overlay> current_overlay_;
		};
	}
}
