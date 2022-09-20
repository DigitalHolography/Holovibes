#pragma once

// FIXME Check for tweakme spdlog : maybe a bad idea ?

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

#define INTERNAL_LOGGER_GET_ARGS_(fmt, ...) __VA_OPT__(, __VA_ARGS__)
#define INTERNAL_LOGGER_GET_ARGS(...) __VA_OPT__(INTERNAL_LOGGER_GET_ARGS_(__VA_ARGS__))

#define LOGGER_PATTERN "%^[%l] [%H:%M:%S.%e] [thread %t]%$ %n >> %v"

#define INTERNAL_LOGGER_GET_FUNC_FMT_(el) #el "={}"
#define INTERNAL_LOGGER_GET_FUNC_FMT(...) FOR_EACH(INTERNAL_LOGGER_GET_FUNC_FMT_, __VA_ARGS__)

#define LOG_TRACE(log, ...) SPDLOG_LOGGER_TRACE(Logger::log(), __VA_ARGS__)
#define LOG_DEBUG(log, ...) SPDLOG_LOGGER_DEBUG(Logger::log(), __VA_ARGS__)
#define LOG_INFO(log, ...) SPDLOG_LOGGER_INFO(Logger::log(), __VA_ARGS__)
#define LOG_WARN(log, ...) SPDLOG_LOGGER_WARN(Logger::log(), __VA_ARGS__)
#define LOG_ERROR(log, ...) SPDLOG_LOGGER_ERROR(Logger::log(), __VA_ARGS__)
#define LOG_CRITICAL(log, ...) SPDLOG_LOGGER_CRITICAL(Logger::log(), __VA_ARGS__)

static constexpr inline const char* const get_file_name(const char* path)
{
    const char* file = path;
    while (*path)
        if (*path++ == '\\')
            file = path;
    return file;
}

#define LOG_FUNC(log, ...)                                                                                             \
    LOG_TRACE(log,                                                                                                     \
              "{}:{} -> {}(" INTERNAL_LOGGER_GET_FUNC_FMT(__VA_ARGS__) ")",                                            \
              get_file_name(__FILE__),                                                                                 \
              __LINE__ INTERNAL_LOGGER_GET_ARGS(log, __FUNCTION__, __VA_ARGS__))

class Logger
{
  public:
    static void init_logger();

    static std::shared_ptr<spdlog::logger> frame_read_worker();
    static std::shared_ptr<spdlog::logger> compute_worker();
    static std::shared_ptr<spdlog::logger> record_worker();
    static std::shared_ptr<spdlog::logger> information_worker();

    static std::shared_ptr<spdlog::logger> cuda();
    static std::shared_ptr<spdlog::logger> setup();
    static std::shared_ptr<spdlog::logger> main();
};

#define INTERNAL_CHECK_GET_FMT()
#define INTERNAL_CHECK_GET_FMT(fmt) fmt
#define INTERNAL_CHECK_GET_FMT(fmt, ...) fmt

#define INTERNAL_CHECK_GET_ARGS()
#define INTERNAL_CHECK_GET_ARGS(fmt)
#define INTERNAL_CHECK_GET_ARGS(fmt, ...) , __VA_ARGS__

#define CHECK(cond, ...)                                                                                               \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            LOG_CRITICAL(main,                                                                                         \
                         "{}:{} " INTERNAL_CHECK_GET_FMT(__VA_ARGS__),                                                 \
                         __FILE__,                                                                                     \
                         __LINE__ INTERNAL_CHECK_GET_ARGS(__VA_ARGS__));                                               \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define CUDA_FATAL(file, line, fmt, ...)                                                                               \
    {                                                                                                                  \
        LOG_CRITICAL(cuda, "{}:{} " fmt, file, line, __VA_ARGS__);                                                     \
        abort();                                                                                                       \
    }
