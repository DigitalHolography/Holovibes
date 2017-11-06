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

namespace holovibes
{
	namespace units
	{

		enum Axis;

		/*! \brief Encapsulates the conversion from a unit to another
		 */
		class ConversionData
		{
		public:
			/*! \brief Constructs an object with the data needed to convert, to be modified for transforms
			 */
			ConversionData(const BasicOpenGLWindow& window);

			/* \brief Converts a unit type into another
			 * {*/
			float window_size_to_opengl(int val, Axis axis) const;
			float fd_to_opengl(int val, Axis axis) const;
			int opengl_to_window_size(float val, Axis axis) const;
			int opengl_to_fd(float val, Axis axis) const;
			/* }
			 */

		private:

			int get_window_size(Axis axis) const;
			int get_fd_size(Axis axis) const;

			const BasicOpenGLWindow&	window_;
		};
	}
}
