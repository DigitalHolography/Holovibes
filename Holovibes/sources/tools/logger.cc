#include "logger.hh"

std::shared_ptr<spdlog::logger> Logger::frame_read_worker()
{
    static auto instance = spdlog::stdout_color_mt("FrameReadWorker");
    return instance;
}

std::shared_ptr<spdlog::logger> Logger::compute_worker()
{
    static auto instance = spdlog::stdout_color_mt("ComputeWorker");
    return instance;
}

std::shared_ptr<spdlog::logger> Logger::record_worker()
{
    static auto instance = spdlog::stdout_color_mt("RecordWorker");
    return instance;
}

std::shared_ptr<spdlog::logger> Logger::information_worker()
{
    static auto instance = spdlog::stdout_color_mt("InformationWorker");
    return instance;
}

std::shared_ptr<spdlog::logger> Logger::cuda()
{
    static auto instance = spdlog::stdout_color_mt("Cuda");
    return instance;
}

std::shared_ptr<spdlog::logger> Logger::setup()
{
    static auto instance = spdlog::stdout_color_mt("Setup");
    return instance;
}

std::shared_ptr<spdlog::logger> Logger::main()
{
    static auto instance = spdlog::stdout_color_mt("Main");
    return instance;
}

// #define LOGGER_PATTERN_OVERRIDE "[%l] [%H:%M:%S.%e] [thread %t] %^%n >> %v%$"

#ifdef LOGGER_PATTERN_OVERRIDE
#undef LOGGER_PATTERN
#define LOGGER_PATTERN LOGGER_PATTERN_OVERRIDE
#endif

void Logger::init_logger()
{
    Logger::frame_read_worker()->set_pattern(LOGGER_PATTERN);
    Logger::compute_worker()->set_pattern(LOGGER_PATTERN);
    Logger::record_worker()->set_pattern(LOGGER_PATTERN);
    Logger::information_worker()->set_pattern(LOGGER_PATTERN);
    Logger::cuda()->set_pattern(LOGGER_PATTERN);
    Logger::setup()->set_pattern(LOGGER_PATTERN);
    Logger::main()->set_pattern(LOGGER_PATTERN);

    Logger::frame_read_worker()->set_level(spdlog::level::trace);
    Logger::compute_worker()->set_level(spdlog::level::trace);
    Logger::record_worker()->set_level(spdlog::level::trace);
    Logger::information_worker()->set_level(spdlog::level::trace);
    Logger::cuda()->set_level(spdlog::level::trace);
    Logger::setup()->set_level(spdlog::level::trace);
    Logger::main()->set_level(spdlog::level::trace);

    spdlog::set_default_logger(Logger::main());
}
