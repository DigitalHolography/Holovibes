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
 * Overlay used to display a reticle in the center of the window. */
#pragma once

#include "BasicOpenGLWindow.hh"
#include "Overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class ReticleOverlay : public Overlay
		{
		public:
			ReticleOverlay(BasicOpenGLWindow* parent);
			virtual ~ReticleOverlay()
			{}

			void init() override;
			void draw() override;

			virtual void move(QMouseEvent *e) override
			{}
			virtual void release(ushort frameside) override
			{}
			virtual void setZone(units::RectFd rect, ushort frameside) override
			{}

		protected:
			void setBuffer() override;

			//! Transparency of the lines
			float alpha_;
		};
	}
}
