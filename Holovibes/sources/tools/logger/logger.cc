#include "logger.hh"
#include "thread_name_flag.hh"
#include "spdlog/pattern_formatter.h"

namespace holovibes
{
std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;
std::unique_ptr<spdlog::pattern_formatter> Logger::formatter_ = nullptr;

std::vector<spdlog::sink_ptr> Logger::sinks_;

std::map<size_t, std::string> Logger::thread_map_;
std::shared_mutex Logger::map_mutex_;

std::shared_ptr<spdlog::logger> Logger::logger()
{
    if (logger_ == nullptr)
    {
        logger_ = init_logger("logger", spdlog::level::trace);
    }
    return logger_;
}

// #define LOGGER_PATTERN_OVERRIDE "[%l] [%H:%M:%S.%e] [thread %t] %^%n >> %v%$"

#ifdef LOGGER_PATTERN_OVERRIDE
#undef LOGGER_PATTERN
#define LOGGER_PATTERN LOGGER_PATTERN_OVERRIDE
#endif

void Logger::init_formatter()
{
    formatter_ = std::make_unique<spdlog::pattern_formatter>();
    formatter_->add_flag<ThreadNameFlag>('t').set_pattern(LOGGER_PATTERN);
}

void Logger::init_sinks()
{
    if (formatter_ == nullptr)
    {
        init_formatter();
    }

    std::filesystem::path log_path = std::filesystem::path(holovibes::settings::logs_dirpath) / "log.txt";
    std::filesystem::path old_log_path = std::filesystem::path(holovibes::settings::logs_dirpath) / "log_prev.txt";

    if (std::filesystem::exists(log_path))
    {
        std::filesystem::rename(log_path, old_log_path);
    }

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path.string(), true);
    file_sink->set_level(spdlog::level::trace);

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);

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
    logger->set_formatter(formatter_->clone());
    return logger;
}

bool Logger::add_thread(std::thread::id thread_id, std::string thread_name)
{
    std::stringstream strstream;
    strstream << thread_id;
    std::string name = thread_name.substr(thread_name.rfind(':') + 1);

    return thread_map_.insert({std::atoll(strstream.str().c_str()), name}).second;
}

std::pair<std::string, bool> Logger::get_thread_name(size_t thread_id)
{
    std::shared_lock lock(map_mutex_);
    std::map<size_t, std::string>::iterator search = thread_map_.find(thread_id);
    if (search == thread_map_.end())
    {
        return {"", false};
    }
    else
    {
        return {search->second, true};
    }
}

} // namespace holovibes
