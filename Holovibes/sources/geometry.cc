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

#include "geometry.hh"

namespace holovibes
{
	Rectangle::Rectangle() : QRect()
	{
	}

	Rectangle::Rectangle(const Rectangle& rect)
		: QRect()
	{
		setTopRight(rect.topRight());
		setTopLeft(rect.topLeft());
		setBottomRight(rect.bottomRight());
		setBottomLeft(rect.bottomLeft());
	}

	Rectangle::Rectangle(const QPoint &topleft, const QSize &size)
		: QRect(topleft, size)
	{
	}

	Rectangle::Rectangle(const unsigned int width, const unsigned int height)
		: QRect(0, 0, width, height)
	{
	}

	unsigned int Rectangle::area() const
	{
		return (width() * height());
	}

	void	Rectangle::checkCorners()
	{
		if (width() < 0)
		{
			QPoint t0pRight = topRight();
			QPoint b0ttomLeft = bottomLeft();

			setTopLeft(t0pRight);
			setBottomRight(b0ttomLeft);
		}
		if (height() < 0)
		{
			QPoint t0pRight = topRight();
			QPoint b0ttomLeft = bottomLeft();

			setTopLeft(b0ttomLeft);
			setBottomRight(t0pRight);
		}
	}

}