#pragma once

#include "logger.hh"

#define CHECK(cond, ...)                                                                                               \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            Logger::main().critical("{}:{} " INTERNAL_LOGGER_GET_FMT(__VA_ARGS__),                                     \
                                    __FILE__,                                                                          \
                                    __LINE__ INTERNAL_LOGGER_GET_ARGS(__VA_ARGS__));                                   \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define CUDA_FATAL(file, line, fmt, ...)                                                                               \
    {                                                                                                                  \
        Logger::cuda().critical("{}:{} " fmt, file, line, __VA_ARGS__);                                                \
        abort();                                                                                                       \
    }
