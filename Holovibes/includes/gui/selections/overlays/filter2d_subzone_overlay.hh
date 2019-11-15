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

/*! \file
*
* Overlay ending the band-pass filtering ROI procedure. */
#pragma once

#include "square_overlay.hh"
#include <memory>
#include "filter2d_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class Filter2DSubZoneOverlay : public SquareOverlay
		{
		public:
			Filter2DSubZoneOverlay(BasicOpenGLWindow* parent);

			void release(ushort frameSide) override;

			void setFilter2dOverlay(std::shared_ptr<Filter2DOverlay> rhs);

		private:
			std::shared_ptr<Filter2DOverlay> filter2d_overlay_;
		};
	}
}