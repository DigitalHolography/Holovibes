#pragma once

#include "logger.hh"

#define INTERNAL_CHECK_GET_FMT()
#define INTERNAL_CHECK_GET_FMT(fmt, ...) fmt

#define INTERNAL_CHECK_GET_ARGS()
#define INTERNAL_CHECK_GET_ARGS(fmt)
#define INTERNAL_CHECK_GET_ARGS(fmt, ...) , __VA_ARGS__

#define CHECK(cond, ...)                                                                                               \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            Logger::main().critical("{}:{} " INTERNAL_CHECK_GET_FMT(__VA_ARGS__),                                      \
                                    __FILE__,                                                                          \
                                    __LINE__ INTERNAL_CHECK_GET_ARGS(__VA_ARGS__));                                    \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define CUDA_FATAL(file, line, fmt, ...)                                                                               \
    {                                                                                                                  \
        Logger::cuda().critical("{}:{} " fmt, file, line, __VA_ARGS__);                                                \
        abort();                                                                                                       \
    }
