
#pragma once
/*! \file
 * Checker macros
 */

#define CHECK_EXPAND(x) x
#define CHECK_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, NAME, ...) NAME

/*! \brief
 * CHECK macro.
 *
 * Support at most 16 arguments
 * usage:
 * CHECK(condition [,formated_string [,args]])
 */

#define CHECK(...)                                                                                                     \
    CHECK_EXPAND(CHECK_GET_MACRO(__VA_ARGS__,                                                                          \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_3,                                                                              \
                                 CHECK_2,                                                                              \
                                 CHECK_1)(__VA_ARGS__))

#define CHECK_1(cond)                                                                                                  \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            LOG_CRITICAL("{}:{} ", __FILE__, __LINE__);                                                                \
            abort();                                                                                                   \
        }                                                                                                              \
    }
#define CHECK_2(cond, fmt)                                                                                             \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            LOG_CRITICAL("{}:{} " fmt, __FILE__, __LINE__);                                                            \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define CHECK_3(cond, fmt, ...)                                                                                        \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            LOG_CRITICAL("{}:{} " fmt, __FILE__, __LINE__, __VA_ARGS__);                                               \
            abort();                                                                                                   \
        }                                                                                                              \
    }