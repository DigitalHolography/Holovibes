#include "camera_logger.hh"

namespace camera
{
std::shared_ptr<spdlog::logger> Logger::camera_ = nullptr;
std::vector<spdlog::sink_ptr> Logger::sinks_;

std::shared_ptr<spdlog::logger> Logger::camera()
{
    if (camera_ == nullptr)
        camera_ = init_logger("Camera", spdlog::level::trace);
    return camera_;
}


#ifdef LOGGER_PATTERN_OVERRIDE
#undef LOGGER_PATTERN
#define LOGGER_PATTERN LOGGER_PATTERN_OVERRIDE
#endif

void Logger::init_sinks()
{
    static auto file_sink =
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(holovibes::settings::logs_dirpath + "/camera_log", true);
    file_sink->set_level(spdlog::level::trace);
    file_sink->set_pattern(LOGGER_PATTERN);

    static auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    console_sink->set_pattern(LOGGER_PATTERN);

    sinks_.push_back(file_sink);
    sinks_.push_back(console_sink);
}

std::shared_ptr<spdlog::logger> Logger::init_logger(std::string name, spdlog::level::level_enum level)
{
    if (sinks_.empty())
    {
        init_sinks();
    }
    auto logger = std::make_shared<spdlog::logger>(name, begin(sinks_), end(sinks_));
    logger->set_level(level);
    return logger;
}
} // namespace camera
