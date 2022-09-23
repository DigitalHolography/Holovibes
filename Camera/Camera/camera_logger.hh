#pragma once

#include "camera.hh"
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "spdlog/fmt/ostr.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#define LOGGER_PATTERN "%^[%=5l] [%H:%M:%S.%e] [thread %t]%$ %n >> %v"

namespace camera
{
static constexpr inline const char* const get_file_name(const char* path)
{
    const char* file = path;
    while (*path)
        if (*path++ == '\\')
            file = path;
    return file;
}

class Logger
{
  public:
    static std::shared_ptr<spdlog::logger> camera();

  private:
    static std::shared_ptr<spdlog::logger> init_logger(std::string name, spdlog::level::level_enum);
    static void init_sinks();

    static std::shared_ptr<spdlog::logger> camera_;
    static std::vector<spdlog::sink_ptr> sinks_;
};
} // namespace camera
