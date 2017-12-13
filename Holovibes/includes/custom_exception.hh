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
* Implementation of custom error class. */
# include <exception>
# include <string>
#pragma once

namespace holovibes
{
	/*! \brief Implementation of custom error class.
	*
	* To create a new kind of error just add your new kind of error to the enum */

	enum error_kind
	{
		fail_update,
		fail_refresh,
		fail_accumulation,
		fail_reference,
		fail_cudaLaunch
	};
	
	class CustomException : public std::exception
	{
	  public:
		CustomException(std::string msg, const error_kind& kind)
			: std::exception(msg.c_str())
			, error_kind_(kind)
		{
		}

	 ~CustomException()
		{
		}

	 const error_kind& get_kind() const
	 {
		 return error_kind_;
	 }

	private :
		
		const error_kind& error_kind_;
			
	};
}