#pragma once

// FIXME Check for tweakme spdlog : maybe a bad idea ?

#define DISABLE_LOG_FUNC
#define DISABLE_LOG_LOCK_MICROCACHE
#define DISABLE_LOG_UPDATE_MAP_ENTRY
// #define DISABLE_LOG_TRIGGER_CACHE
// #define DISABLE_LOG_TRIGGER_REF
// #define DISABLE_LOG_GSH_ON_CHANGE
// #define DISABLE_LOG_SYNC_MICROCACHE
// #define DISABLE_LOG_PIPE
// #define DISABLE_LOG_UPDATE_ON_SYNC
// #define DISABLE_LOG_UPDATE_FRONT_END
// #define DISABLE_LOG_UPDATE_ON_CHANGE

#ifndef DISABLE_LOG_UPDATE_ON_SYNC
#define LOG_UPDATE_ON_SYNC(type) LOG_TRACE("UPDATE OnSync : " #type);
#else
#define LOG_UPDATE_ON_SYNC(type)
#endif

#ifndef DISABLE_LOG_UPDATE_FRONT_END
#define LOG_UPDATE_FRONT_END_BEFORE(type) LOG_TRACE("UPDATE FrontEnd before : " #type);
#define LOG_UPDATE_FRONT_END_AFTER(type) LOG_TRACE("UPDATE FrontEnd after : " #type);
#else
#define LOG_UPDATE_FRONT_END_BEFORE(type)
#define LOG_UPDATE_FRONT_END_AFTER(type)
#endif

#ifndef DISABLE_LOG_UPDATE_ON_CHANGE
#define LOG_UPDATE_ON_CHANGE(type) LOG_TRACE("UPDATE OnChange : " #type);
#else
#define LOG_UPDATE_ON_CHANGE(type)
#endif

#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "check.hh"
#include "holovibes_config.hh"
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#define LOGGER_PATTERN "%^[%=5l] [%H:%M:%S.%e] [%t] [%N] %$ %v"

#define LOG_TRACE(fmt, ...)                                                                                            \
    SPDLOG_LOGGER_TRACE(holovibes::Logger::logger(), "{}:{} " fmt, get_file_name(__FILE__), __LINE__, __VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(holovibes::Logger::logger(), __VA_ARGS__)
#define LOG_INFO(...) SPDLOG_LOGGER_INFO(holovibes::Logger::logger(), __VA_ARGS__)
#define LOG_WARN(...) SPDLOG_LOGGER_WARN(holovibes::Logger::logger(), __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_ERROR(holovibes::Logger::logger(), __VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(holovibes::Logger::logger(), __VA_ARGS__)

#define CUDA_FATAL(file, line, fmt, ...)                                                                               \
    {                                                                                                                  \
        LOG_CRITICAL("{}:{} " fmt, file, line, __VA_ARGS__);                                                           \
        abort();                                                                                                       \
    }

#ifndef DISABLE_LOG_FUNC
#define LOG_FUNC(...) LOG_TRACE("-> {}(" INTERNAL_LOGGER_GET_FUNC_FMT(__VA_ARGS__) ")", __FUNCTION__, __VA_ARGS__)
#else
#define LOG_FUNC(...)
#endif

constexpr const char* get_file_name(const char* path)
{
    const char* file = path;
    while (*path)
        if (*path++ == '\\')
            file = path;
    return file;
}

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

#define INTERNAL_LOGGER_GET_FUNC_FMT_(el) #el "={}"
#define INTERNAL_LOGGER_GET_FUNC_FMT(...) FOR_EACH(INTERNAL_LOGGER_GET_FUNC_FMT_, __VA_ARGS__)

namespace holovibes
{
class Logger
{

  public:
    static std::shared_ptr<spdlog::logger> logger();
    static bool add_thread(std::thread::id thread_id, std::string thread_name);
    static std::pair<std::string, bool> get_thread_name(size_t thread_id);
    static void flush();

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
