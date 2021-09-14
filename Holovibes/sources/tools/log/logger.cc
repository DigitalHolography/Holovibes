#include "logger.hh"

Logger* Logger::instance()
{
    static Logger logger;
    return &logger;
}

Logger::Logger()
{
#ifdef _DEBUG
    set_level(loguru::Verbosity_MAX);
#else
    set_level(loguru::Verbosity_INFO);
#endif
}

LoggingLine Logger::logging__(loguru::NamedVerbosity level)
{
    return LoggingLine(level);
}

void Logger::set_level(loguru::NamedVerbosity level)
{
    loguru::g_stderr_verbosity = level;
}
