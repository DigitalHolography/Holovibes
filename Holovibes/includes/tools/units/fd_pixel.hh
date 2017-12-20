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
 * Implementation of a position in the frame desc coordinate system */
#pragma once

#include "unit.hh"

namespace holovibes
{
	namespace units
	{
		class WindowPixel;
		class OpenglPosition;
		class RealPosition;

		/*! \brief A position in the frame desc coordinate system [0;fd.width]
		 */
		class FDPixel : public Unit<int>
		{
		public:
			FDPixel(ConversionData data, Axis axis, int val = 0);

			operator OpenglPosition() const;
			operator WindowPixel() const;
			operator RealPosition() const;
		};
	}
}
