#include "logger2.hh"

#include "loguru.hpp"

Logger2 *Logger2::instance()
{
    static Logger2 logger;
    return &logger;
}

Logger2::Logger2()
{
#ifdef _DEBUG
    set_level(loguru::Verbosity_MAX);
#else
    set_level(loguru::Verbosity_INFO);
#endif
}

LoggingLine Logger2::logging__(loguru::NamedVerbosity level)
{
    return LoggingLine(level);
}

void Logger2::set_level(loguru::NamedVerbosity level)
{
    loguru::g_stderr_verbosity = level;
}