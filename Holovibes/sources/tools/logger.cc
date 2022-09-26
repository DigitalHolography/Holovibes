#include "logger.hh"

namespace holovibes
{
std::shared_ptr<spdlog::logger> Logger::frame_read_worker_ = nullptr;
std::shared_ptr<spdlog::logger> Logger::compute_worker_ = nullptr;
std::shared_ptr<spdlog::logger> Logger::record_worker_ = nullptr;
std::shared_ptr<spdlog::logger> Logger::information_worker_ = nullptr;
std::shared_ptr<spdlog::logger> Logger::cuda_ = nullptr;
std::shared_ptr<spdlog::logger> Logger::setup_ = nullptr;
std::shared_ptr<spdlog::logger> Logger::main_ = nullptr;
std::vector<spdlog::sink_ptr> Logger::sinks_;

std::shared_ptr<spdlog::logger> Logger::frame_read_worker()
{
    if (frame_read_worker_ == nullptr)
    {
        frame_read_worker_ = init_logger("FrameReadWorker", spdlog::level::trace);
    }
    return frame_read_worker_;
}

std::shared_ptr<spdlog::logger> Logger::compute_worker()
{
    if (compute_worker_ == nullptr)
    {
        compute_worker_ = init_logger("ComputeWorker", spdlog::level::trace);
    }
    return compute_worker_;
}

std::shared_ptr<spdlog::logger> Logger::record_worker()
{
    if (record_worker_ == nullptr)
    {
        record_worker_ = init_logger("RecordWorker", spdlog::level::trace);
    }
    return record_worker_;
}

std::shared_ptr<spdlog::logger> Logger::information_worker()
{
    if (information_worker_ == nullptr)
    {
        information_worker_ = init_logger("InformationLogger", spdlog::level::trace);
    }
    return information_worker_;
}

std::shared_ptr<spdlog::logger> Logger::cuda()
{
    if (cuda_ == nullptr)
    {
        cuda_ = init_logger("Cuda", spdlog::level::trace);
    }
    return cuda_;
}

std::shared_ptr<spdlog::logger> Logger::setup()
{
    if (setup_ == nullptr)
    {
        setup_ = init_logger("Setup", spdlog::level::trace);
    }
    return setup_;
}

std::shared_ptr<spdlog::logger> Logger::main()
{
    if (main_ == nullptr)
    {
        main_ = init_logger("Main", spdlog::level::trace);
    }
    return main_;
}

// #define LOGGER_PATTERN_OVERRIDE "[%l] [%H:%M:%S.%e] [thread %t] %^%n >> %v%$"

#ifdef LOGGER_PATTERN_OVERRIDE
#undef LOGGER_PATTERN
#define LOGGER_PATTERN LOGGER_PATTERN_OVERRIDE
#endif

void Logger::init_sinks()
{
    static auto file_sink =
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(holovibes::settings::logs_dirpath + "/log", true);
    file_sink->set_level(spdlog::level::trace);
    file_sink->set_pattern(LOGGER_PATTERN);

    static auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
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

} // namespace holovibes
