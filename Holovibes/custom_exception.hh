# include <exception>
# include <string>
/*! \file
*
* Implementation of the Observer design pattern. */
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
		fail_reference
	};
	
	class CustomException : public std::exception
	{
	  public:
		CustomException(std::string msg, const error_kind& kind)
			: std::exception()
			, error_msg_(msg)
			, error_kind_(kind)
		{
		}

	 ~CustomException()
		{
		}

	 const char* what() const throw()
	 {
		 return error_msg_.c_str();
	 };

	 const error_kind& get_kind()
	 {
		 return error_kind_;
	 }


	private :

		const std::string& error_msg_;
		
		const error_kind& error_kind_;
			
	};
}