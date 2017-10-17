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

#include "BasicOpenGLWindow.hh"
#include "Overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class CrossOverlay : public Overlay
		{
		public:
			CrossOverlay(KindOfView view, BasicOpenGLWindow* parent);
			virtual ~CrossOverlay();

			void setBuffer(QPoint pos, QSize frame);
			void setDoubleBuffer(QPoint pos1, QPoint pos2, QSize frame);

			void init() override;
			void draw() override;

			// Not called when using cross
			void move(QPoint pos, QSize win_size) override
			{}

			// Not called when using cross
			void release(ushort frameSide) override
			{}

		private:
			void drawCross(GLuint offset, GLsizei count);

			bool doubleCross_;
		};
	}
}
