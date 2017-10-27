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

#include <ostream>
#include <qrect.h>

namespace holovibes
{
	namespace gui
	{
		enum KindOfOverlay
		{
			Zoom,
			// Average
			Signal,
			Noise,
			// -------
			Autofocus,
			Filter2D,
			SliceZoom,
			Cross,
			SliceCross,
			Strip
		};
		class Rectangle : public QRect
		{
		public:
			Rectangle();
			Rectangle(const QRect& rect);
			Rectangle(const Rectangle& rect);
			Rectangle(const QPoint &topleft, const QSize &size);
			Rectangle(const uint width, const uint height);
			
			uint	area() const;
		};
		std::ostream& operator<<(std::ostream& os, const Rectangle& obj);
		Rectangle operator-(Rectangle& rec, const QPoint& point);
	}
	std::ostream& operator<<(std::ostream& os, const QPoint& p);
	std::ostream& operator<<(std::ostream& os, const QSize& s);
}
