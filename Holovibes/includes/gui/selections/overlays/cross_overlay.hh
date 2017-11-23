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
 * Overlay used to compute the side views. */
#pragma once

#include "BasicOpenGLWindow.hh"
#include "Overlay.hh"
#include "zoom_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class CrossOverlay : public Overlay
		{
		public:
			CrossOverlay(BasicOpenGLWindow* parent);
			virtual ~CrossOverlay();

			/*! \brief Initialize opengl buffers for rectangles and lines.
			 *  The vertices buffers is built like this:
			 *
			 *       0   1
			 *       |   |
			 *  4 --- --- --- 5
			 *       |   |
			 *  7 --- --- --- 6
			 *       |   |
			 *       3   2
			 */
			void init() override;
			void draw() override;

			void press(QMouseEvent *e) override;
			void keyPress(QKeyEvent *e) override;
			void move(QMouseEvent *e) override;
			void release(ushort frameSide) override;

			// Not called when using cross
			void setZone(units::RectFd rect, ushort frameside) override
			{}

		protected:
			void setBuffer() override;

			/*! \brief Computes the zones depending on compute descriptor of the parent */
			void computeZone();

			//! Transparency of the borders
			float line_alpha_;

			//! Vertices order for lines
			GLuint elemLineIndex_;

			//! Locking line overlay
			bool locked_;

			//! Actual mouse position
			units::PointFd mouse_position_;

			//! Horizontal area. zone_ corresponds to the vertical area
			units::RectFd horizontal_zone_;
		};
	}
}
