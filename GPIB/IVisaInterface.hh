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

#include <string>
#include <optional>

#include "batch_command.hh"

/*! \brief contains all the functions used in the GPIB project */
namespace gpib
{
	/*! \brief Interface in order to load the dll runtime
	 *
	 * We want holovibes to run even if the computer doesn't have visa
	 * installed on it (it is very heavy and long to install). So what we
	 * want is to load visa's dll only when needed (batch input).
	 *
	 * This interface only has got these methods because they are the one
	 * used in holovibes. The rest of the methods are only used and GPIB
	 * and don't need to be in there.
	 *
	 * Why do we use this interface ?
	 *
	 * Because otherwhise, holovibes would be faced to some unrecognized symbols
	 */
	class IVisaInterface
	{
	public:
		IVisaInterface()
		{}

		virtual ~IVisaInterface()
		{}

		virtual void execute_instrument_command(const BatchCommand& instrument_command) = 0;
	};

	/* \brief See icamera.hh to have more information about this */
	extern "C"
	{
		__declspec(dllexport) IVisaInterface* new_gpib_controller();
	}
}
