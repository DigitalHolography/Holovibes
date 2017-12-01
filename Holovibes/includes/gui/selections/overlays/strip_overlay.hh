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
* Overlay displaying a strip of color in function of the parent window. */
#pragma once

#include "rect_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class StripOverlay : public RectOverlay
		{
		public:
			StripOverlay(BasicOpenGLWindow* parent,
				Component& component,
				std::atomic<ushort>& nsamples,
				Color color,
				float alpha = 0.3f);

			void release(ushort frameSide) override
			{}

			void move(QMouseEvent *e) override
			{}

			void draw() override;

			void compute_zone();

		private:
			Component& component_;
			std::atomic<ushort>& nsamples_;
		};
	}
}