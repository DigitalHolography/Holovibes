#pragma once

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <string>

#include <exception>
#include <fstream>
#include <sstream>
#include <functional>
#include <iostream>

// FROM : https://www.scs.stanford.edu/~dm/blog/va-opt.html
#define PARENS ()
#define EXPAND(arg) EXPAND1(EXPAND1(EXPAND1(EXPAND1(arg))))
#define EXPAND1(arg) EXPAND2(EXPAND2(EXPAND2(EXPAND2(arg))))
#define EXPAND2(arg) EXPAND3(EXPAND3(EXPAND3(EXPAND3(arg))))
#define EXPAND3(arg) EXPAND4(EXPAND4(EXPAND4(EXPAND4(arg))))
#define EXPAND4(arg) arg
#define FOR_EACH(macro, ...) __VA_OPT__(EXPAND(FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define FOR_EACH_HELPER(macro, a1, ...) macro(a1) __VA_OPT__(", " FOR_EACH_AGAIN PARENS(macro, __VA_ARGS__))
#define FOR_EACH_AGAIN() FOR_EACH_HELPER
// STOP : https://www.scs.stanford.edu/~dm/blog/va-opt.html

#define INTERNAL_LOGGER_GET_FMT()
#define INTERNAL_LOGGER_GET_FMT(fmt, ...) fmt

#define INTERNAL_LOGGER_GET_ARGS()
#define INTERNAL_LOGGER_GET_ARGS(fmt)
#define INTERNAL_LOGGER_GET_ARGS(fmt, ...) , __VA_ARGS__

#define LOGGER_PATTERN "[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v"

#define INTERNAL_LOGGER_GET_FUNC_FMT_()
#define INTERNAL_LOGGER_GET_FUNC_FMT_(el) #el "={}"
#define INTERNAL_LOGGER_GET_FUNC_FMT(...) FOR_EACH(INTERNAL_LOGGER_GET_FUNC_FMT_, __VA_ARGS__)

#define LOG_FUNC(log, ...)                                                                                             \
    Logger::log().trace(                                                                                               \
        "{}(" INTERNAL_LOGGER_GET_FUNC_FMT(__VA_ARGS__) ")" INTERNAL_LOGGER_GET_ARGS(log, __FUNCTION__, __VA_ARGS__))

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
