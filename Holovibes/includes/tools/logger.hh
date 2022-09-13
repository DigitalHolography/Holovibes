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
    static void init_logger(bool debug_mode);

    static spdlog::logger& frame_read_worker();
    static spdlog::logger& compute_worker();
    static spdlog::logger& record_worker();
    static spdlog::logger& information_worker();

    static spdlog::logger& cuda();
    static spdlog::logger& setup();

    static spdlog::logger& fatal();
    static spdlog::logger& trace();
    static spdlog::logger& main();
    static std::shared_ptr<spdlog::logger> main_ptr();
};

#define LOG_TRACE(...) Logger::main().trace(__VA_ARGS__)
#define LOG_INFO(...) Logger::main().info(__VA_ARGS__)
#define LOG_WARN(...) Logger::main().warn(__VA_ARGS__)
#define LOG_ERROR(...) Logger::main().error(__VA_ARGS__)
#define LOG_CRITICAL(...) Logger::main().critical(__VA_ARGS__)

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
