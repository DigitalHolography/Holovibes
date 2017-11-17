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
* Overlay displaying the scale of the image. Could be factorized with slicecross overlay and maybe with strip */
#pragma once

#include "rect_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class ScaleOverlay : public RectOverlay
		{
		public:
			ScaleOverlay(BasicOpenGLWindow* parent);
			~ScaleOverlay();

			void init() override;
			void draw() override;

			void keyPress(QKeyEvent *e) override
			{ }

			void move(QMouseEvent *e) override
			{ }

			void release(ushort frameSide) override
			{ }

			void setBuffer() override;
		private:
			//! Transparency of the borders
			float line_alpha_;

			//! Vertices order for lines
			GLuint elemLineIndex_;
		};
	}
}
