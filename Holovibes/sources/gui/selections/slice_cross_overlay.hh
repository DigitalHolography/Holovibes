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

#include "rect_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class SliceCrossOverlay : public RectOverlay
		{
		public:
			SliceCrossOverlay(BasicOpenGLWindow* parent);

			void init() override;
			void draw() override;

			void keyPress(QPoint pos) override;
			void move(QPoint pos) override;
			void release(ushort frameSide) override;

			void setBuffer();
		private:
			//! Transparency of the borders
			float line_alpha_;

			//! Vertices order for lines
			GLuint elemLineIndex_;

			//! Locking line overlay
			bool locked_;

			//! Last click
			QPoint last_clicked_;
		};
	}
}