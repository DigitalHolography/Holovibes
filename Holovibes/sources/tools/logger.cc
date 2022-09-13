#include "logger.hh"

spdlog::logger& Logger::frame_read_worker()
{
    static auto instance = spdlog::stdout_color_mt("frame_read_worker");
    return *instance;
}

spdlog::logger& Logger::compute_worker()
{
    static auto instance = spdlog::stdout_color_mt("compute_worker");
    return *instance;
}

spdlog::logger& Logger::record_worker()
{
    static auto instance = spdlog::stdout_color_mt("record_worker");
    return *instance;
}

spdlog::logger& Logger::information_worker()
{
    static auto instance = spdlog::stdout_color_mt("information_worker");
    return *instance;
}

spdlog::logger& Logger::cuda()
{
    static auto instance = spdlog::stdout_color_mt("cuda");
    return *instance;
}

spdlog::logger& Logger::fatal()
{
    static auto instance = spdlog::stdout_color_mt("fatal");
    return *instance;
}

spdlog::logger& Logger::trace()
{
    static auto instance = spdlog::stdout_color_mt("trace");
    return *instance;
}

spdlog::logger& Logger::logger()
{
    static auto instance = spdlog::stdout_color_mt("logger");
    return *instance;
}
