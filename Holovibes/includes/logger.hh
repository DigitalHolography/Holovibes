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
		std::cout << "(FILE: " << get_file_name(path) << " / LINE: " << line << " / FUNCTION: " << func << ")" << "\n";
#endif
		std::cout << msg << "\n";
#ifdef _DEBUG
		std::cout << "\n";
#endif

	}

private:
	static std::string get_file_name(const std::string& path)
	{
		size_t index = path.find_last_of('\\');
		return path.substr(index == path.npos ? 0 : index + 1);
	}
};