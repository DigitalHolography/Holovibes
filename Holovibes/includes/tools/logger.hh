#pragma once

#include <string>

#include <exception>
#include <fstream>
#include <sstream>
#include <functional>
#include <iostream>

#define LOGURU_WITH_STREAMS 1

#include <loguru.hpp>

#define LOG_TRACE LOG_S(2)
#define LOG_DEBUG LOG_S(1)
#define LOG_INFO LOG_S(INFO)
#define LOG_WARN LOG_S(WARNING)
#define LOG_ERROR LOG_S(ERROR)
#define LOG_FUNC LOG_SCOPE_FUNCTION(1)

#define catch_log(e) catch_log__((e), __LINE__, __FILE__)

#ifdef _DEBUG

inline void catch_log__(const std::exception& e, int line, const char* file)
{
    LOG_ERROR << "Internal Error occured: " << e.what();
    LOG_ERROR << "Error occured in file " << file << " at line " << line;
    throw e;
}

#else

inline void catch_log__(const std::exception& e, int, const char*)
{
    LOG_ERROR << "Internal Error occured: " << e.what() << '\n';
}

#endif
