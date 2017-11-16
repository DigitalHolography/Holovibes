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
 * Interface for all square overlays. */
#pragma once

#include "rect_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class SquareOverlay : public RectOverlay
		{
		public:
			SquareOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);

			virtual ~SquareOverlay()
			{}

			/*! \brief Check if corners don't go out of bounds. */
			void checkCorners();
			/*! \brief Change the rectangular zone to a square zone, using the shortest side */
			void make_square();

			virtual void move(QMouseEvent *e) override;
		};
	}
}
