#pragma once

#include <string>
#define LOGURU_WITH_STREAMS 1

#include <exception>
#include <fstream>
#include <functional>
#include <iostream>

#include "loguru.hpp"

#define LOG_TRACE (logging(2))
#define LOG_DEBUG (logging(1))
#define LOG_INFO (logging(INFO))
#define LOG_WARN (logging(WARNING))
#define LOG_ERROR (logging(ERROR))

#define logging(lvl) Logger::instance()->logging__(loguru::Verbosity_##lvl)

#define catch_log(e) catch_log__((e), __LINE__, __FILE__)

class LoggingLine
{
  public:
    LoggingLine(loguru::NamedVerbosity level)
        : level_(level)
    {
    }

    void add_data(const std::string& s) { line += s; }

    void add_data(const char& s) { line += s; }

    void add_data(const char* s) { line += s; }

    template <typename T>
    void add_data(const T& s)
    {
        line += std::to_string(s);
    }

    template <typename T>
    LoggingLine& operator<<(const T& data)
    {
        add_data(data);
        return *this;
    }

    template <typename CharT, typename Traits>
    LoggingLine& operator<<(const std::basic_ostream<CharT, Traits>& (*endl)(
        std::basic_ostream<CharT, Traits>&)
    {
        add_data(std::endl);
        return *this;
    }

    ~LoggingLine()
    {
        if (!line.empty())
            VLOG_F(level_, "%s", line.c_str());
    }

  private:
    loguru::NamedVerbosity level_;
    std::string line;
};

class Logger
{
  public:
    static Logger* instance();

    void set_level(loguru::NamedVerbosity);

    LoggingLine logging__(loguru::NamedVerbosity level);

  private:
    Logger();
};

#ifdef _DEBUG

inline void catch_log__(const std::exception& e, int line, const char* file)
{
    LOG_ERROR << "Internal Error occured: " << e.what();
    LOG_ERROR << "Error occured in file " << file << " at line " << line;
    throw e;
}

#else

inline void catch_log__(const std::exception& e, int, const char*)
{
    LOG_ERROR << "Internal Error occured: " << e.what() << '\n';
}

#endif
