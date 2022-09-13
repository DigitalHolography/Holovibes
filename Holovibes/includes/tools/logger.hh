#pragma once

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <string>

#include <exception>
#include <fstream>
#include <sstream>
#include <functional>
#include <iostream>

class Logger
{
  public:
    static spdlog::logger& frame_read_worker();
    static spdlog::logger& compute_worker();
    static spdlog::logger& record_worker();
    static spdlog::logger& information_worker();

    static spdlog::logger& cuda();
    static spdlog::logger& fatal();
    static spdlog::logger& trace();
    static spdlog::logger& logger();
};

#define LOG_TRACE
#define LOG_DEBUG
#define LOG_INFO
#define LOG_WARN
#define LOG_ERROR
#define LOG_FUNC

#define catch_log(e) catch_log__((e), __LINE__, __FILE__)

#ifdef _DEBUG

inline void catch_log__(const std::exception& e, int line, const char* file)
{
    // LOG_ERROR << "Internal Error occured: " << e.what();
    // LOG_ERROR << "Error occured in file " << file << " at line " << line;
    throw e;
}

#else

inline void catch_log__(const std::exception& e, int, const char*)
{
    // LOG_ERROR << "Internal Error occured: " << e.what() << '\n';
}

#endif
