#pragma once

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <string>

#include <exception>
#include <fstream>
#include <sstream>
#include <functional>
#include <iostream>

#define LOGGER_PATTERN "[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v"

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

    static spdlog::logger& api();

    static spdlog::logger& main();
    static std::shared_ptr<spdlog::logger> main_ptr();
};
