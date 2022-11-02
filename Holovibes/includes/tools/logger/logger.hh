#pragma once

// FIXME Check for tweakme spdlog : maybe a bad idea ?

#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <shared_mutex>

#include "holovibes_config.hh"
#include "spdlog/fmt/ostr.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "check.hh"

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

#define INTERNAL_LOGGER_GET_FUNC_FMT_(el) #el "={}"
#define INTERNAL_LOGGER_GET_FUNC_FMT(...) FOR_EACH(INTERNAL_LOGGER_GET_FUNC_FMT_, __VA_ARGS__)

#define LOG_TRACE(log, ...) SPDLOG_LOGGER_TRACE(holovibes::Logger::logger(), __VA_ARGS__)
#define LOG_DEBUG(log, ...) SPDLOG_LOGGER_DEBUG(holovibes::Logger::logger(), __VA_ARGS__)
#define LOG_INFO(log, ...) SPDLOG_LOGGER_INFO(holovibes::Logger::logger(), __VA_ARGS__)
#define LOG_WARN(log, ...) SPDLOG_LOGGER_WARN(holovibes::Logger::logger(), __VA_ARGS__)
#define LOG_ERROR(log, ...) SPDLOG_LOGGER_ERROR(holovibes::Logger::logger(), __VA_ARGS__)
#define LOG_CRITICAL(log, ...) SPDLOG_LOGGER_CRITICAL(holovibes::Logger::logger(), __VA_ARGS__)

constexpr const char* get_file_name(const char* path)
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

#define CUDA_FATAL(file, line, fmt, ...)                                                                               \
    {                                                                                                                  \
        LOG_CRITICAL(cuda, "{}:{} " fmt, file, line, __VA_ARGS__);                                                     \
        abort();                                                                                                       \
    }

#define LOGGER_PATTERN "%^[%=5l] [%H:%M:%S.%e] [%t]%$ %v"

namespace holovibes
{
enum class State
{
    free,
    read,
    write
};

class Logger
{

  public:
    static std::shared_ptr<spdlog::logger> logger();
    static bool add_thread(std::thread::id thread_id, std::string thread_name);
    static std::pair<std::string, bool> get_thread_name(size_t thread_id);
    static std::shared_mutex map_mutex_;

  private:
    static std::shared_ptr<spdlog::logger> init_logger(std::string name, spdlog::level::level_enum);
    static void init_sinks();
    static void init_formatter();

    static std::shared_ptr<spdlog::logger> logger_;

    static std::unique_ptr<spdlog::pattern_formatter> formatter_;

    static std::vector<spdlog::sink_ptr> sinks_;

    static std::map<size_t, std::string> thread_map_;
};

} // namespace holovibes
