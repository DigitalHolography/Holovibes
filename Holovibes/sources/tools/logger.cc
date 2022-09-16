#include "logger.hh"

spdlog::logger& Logger::frame_read_worker()
{
    static auto instance = spdlog::stdout_color_mt("FrameReadWorker");
    return *instance;
}

spdlog::logger& Logger::compute_worker()
{
    static auto instance = spdlog::stdout_color_mt("ComputeWorker");
    return *instance;
}

spdlog::logger& Logger::record_worker()
{
    static auto instance = spdlog::stdout_color_mt("RecordWorker");
    return *instance;
}

spdlog::logger& Logger::information_worker()
{
    static auto instance = spdlog::stdout_color_mt("InformationWorker");
    return *instance;
}

spdlog::logger& Logger::cuda()
{
    static auto instance = spdlog::stdout_color_mt("Cuda");
    return *instance;
}

spdlog::logger& Logger::setup()
{
    static auto instance = spdlog::stdout_color_mt("Setup");
    return *instance;
}

spdlog::logger& Logger::api()
{
    static auto instance = spdlog::stdout_color_mt("API");
    return *instance;
}

spdlog::logger& Logger::main() { return *main_ptr(); }

std::shared_ptr<spdlog::logger> Logger::main_ptr()
{
    static auto instance = spdlog::stdout_color_mt("Main");
    return instance;
}

void Logger::init_logger([[maybe_unused]] bool debug_mode)
{
    Logger::frame_read_worker().set_pattern(LOGGER_PATTERN);
    Logger::compute_worker().set_pattern(LOGGER_PATTERN);
    Logger::record_worker().set_pattern(LOGGER_PATTERN);
    Logger::information_worker().set_pattern(LOGGER_PATTERN);
    Logger::cuda().set_pattern(LOGGER_PATTERN);
    Logger::setup().set_pattern(LOGGER_PATTERN);
    Logger::api().set_pattern(LOGGER_PATTERN);
    Logger::main().set_pattern(LOGGER_PATTERN);

    Logger::frame_read_worker().set_level(spdlog::level::trace);
    Logger::compute_worker().set_level(spdlog::level::trace);
    Logger::record_worker().set_level(spdlog::level::trace);
    Logger::information_worker().set_level(spdlog::level::trace);
    Logger::cuda().set_level(spdlog::level::trace);
    Logger::setup().set_level(spdlog::level::trace);
    Logger::api().set_level(spdlog::level::trace);
    Logger::main().set_level(spdlog::level::trace);

    spdlog::set_default_logger(Logger::main_ptr());
}
