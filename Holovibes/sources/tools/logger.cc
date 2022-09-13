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

spdlog::logger& Logger::fatal()
{
    static auto instance = spdlog::stdout_color_mt("Fatal");
    return *instance;
}

spdlog::logger& Logger::trace()
{
    static auto instance = spdlog::stdout_color_mt("Trace");
    return *instance;
}

spdlog::logger& Logger::setup()
{
    static auto instance = spdlog::stdout_color_mt("Setup");
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
    Logger::frame_read_worker().set_pattern("[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v");
    Logger::compute_worker().set_pattern("[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v");
    Logger::record_worker().set_pattern("[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v");
    Logger::information_worker().set_pattern("[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v");

    Logger::cuda().set_pattern("[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v");
    Logger::setup().set_pattern("[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v");

    Logger::fatal().set_pattern("[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v");
    Logger::trace().set_pattern("[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v");
    Logger::main().set_pattern("[%^%l%$] [%H:%M:%S.%e] [thread %t] %n >> %v");
    spdlog::set_default_logger(Logger::main_ptr());
}
