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

#include <iostream>
#include <string>

#define LOG_MSG(Msg) Logger::log((Msg), __FILE__, __LINE__, __FUNCTION__ )
#define LOG_INFO(Msg) Logger::log(std::string("[INFO] ") + (Msg), __FILE__, __LINE__, __FUNCTION__ )
#define LOG_WARN(Msg) Logger::log(std::string("[WARNING] ") + (Msg), __FILE__, __LINE__, __FUNCTION__ )
#define LOG_ERROR(Msg) Logger::log(std::string("[ERROR] ") + (Msg), __FILE__, __LINE__, __FUNCTION__ )

class Logger
{
public:
	static void log(const std::string& msg, const std::string& path, int line, const std::string& func)
	{
#ifdef _DEBUG
		std::cout << "(FILE: " << get_file_name(path) << " / LINE: " << line << " / FUNCTION: " << func << ")" << std::endl;
#endif
		std::cout << msg << std::endl;
#ifdef _DEBUG
		std::cout << std::endl;
#endif

	}

private:
	static std::string get_file_name(const std::string& path)
	{
		size_t index = path.find_last_of('\\');
		return path.substr(index == path.npos ? 0 : index + 1);
	}
};