
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
 *  Enum for the different type of displaying images
 */
#pragma once

namespace holovibes
{
	/*! \brief	Displaying type of the image */
	enum class ImgType
	{
		Modulus = 0,/*!< Modulus of the complex data */
		SquaredModulus,/*!<  Modulus taken to its square value */
		Argument,/*!<  Phase (angle) value of the complex pixel c, computed with atan(Im(c)/Re(c)) */
		PhaseIncrease,/*!<  Phase value computed with the conjugate between the phase of the last image and the previous one */
		Composite/*!<  Displays different frequency intervals on color RBG or HSV chanels*/
	};
} // namespace holovibes