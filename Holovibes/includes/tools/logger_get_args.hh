/*! \file
 * macros to get all but the first argument
 */

/*
 * TODO: the day __VA_OPT__ is usable, this file can be replaced by 2 lines:
 *
 * #define INTERNAL_CHECK_GET_ARGS_(fmt, ...) __VA_OPT__(, ) __VA_ARGS__
 * #define INTERNAL_CHECK_GET_ARGS(...) __VA_OPT__(INTERNAL_CHECK_GET_ARGS_(__VA_ARGS__))
 */

#define INTERNAL_CHECK_GET_ARGS_EXPAND(x) x
#define INTERNAL_CHECK_GET_ARGS_GET_MACRO(_1,                                                                          \
                                          _2,                                                                          \
                                          _3,                                                                          \
                                          _4,                                                                          \
                                          _5,                                                                          \
                                          _6,                                                                          \
                                          _7,                                                                          \
                                          _8,                                                                          \
                                          _9,                                                                          \
                                          _10,                                                                         \
                                          _11,                                                                         \
                                          _12,                                                                         \
                                          _13,                                                                         \
                                          _14,                                                                         \
                                          _15,                                                                         \
                                          _16,                                                                         \
                                          NAME,                                                                        \
                                          ...)                                                                         \
    NAME

/*! \brief
 * macro to be used
 * at most 16 arguments
 */
#define INTERNAL_CHECK_GET_ARGS(...)                                                                                   \
    INTERNAL_CHECK_GET_ARGS_EXPAND(INTERNAL_CHECK_GET_ARGS_GET_MACRO(__VA_ARGS__,                                      \
                                                                     INTERNAL_CHECK_GET_ARGS_16,                       \
                                                                     INTERNAL_CHECK_GET_ARGS_15,                       \
                                                                     INTERNAL_CHECK_GET_ARGS_14,                       \
                                                                     INTERNAL_CHECK_GET_ARGS_13,                       \
                                                                     INTERNAL_CHECK_GET_ARGS_12,                       \
                                                                     INTERNAL_CHECK_GET_ARGS_11,                       \
                                                                     INTERNAL_CHECK_GET_ARGS_10,                       \
                                                                     INTERNAL_CHECK_GET_ARGS_9,                        \
                                                                     INTERNAL_CHECK_GET_ARGS_8,                        \
                                                                     INTERNAL_CHECK_GET_ARGS_7,                        \
                                                                     INTERNAL_CHECK_GET_ARGS_6,                        \
                                                                     INTERNAL_CHECK_GET_ARGS_5,                        \
                                                                     INTERNAL_CHECK_GET_ARGS_4,                        \
                                                                     INTERNAL_CHECK_GET_ARGS_3,                        \
                                                                     INTERNAL_CHECK_GET_ARGS_2,                        \
                                                                     INTERNAL_CHECK_GET_ARGS_1)(__VA_ARGS__))

#define INTERNAL_CHECK_GET_ARGS_1(fmt)
#define INTERNAL_CHECK_GET_ARGS_2(fmt, v1) , v1
#define INTERNAL_CHECK_GET_ARGS_3(fmt, v1, v2) , v1, v2
#define INTERNAL_CHECK_GET_ARGS_4(fmt, v1, v2, v3) , v1, v2, v3
#define INTERNAL_CHECK_GET_ARGS_5(fmt, v1, v2, v3, v4) , v1, v2, v3, v4
#define INTERNAL_CHECK_GET_ARGS_6(fmt, v1, v2, v3, v4, v5) , v1, v2, v3, v4, v5
#define INTERNAL_CHECK_GET_ARGS_7(fmt, v1, v2, v3, v4, v5, v6) , v1, v2, v3, v4, v5, v6
#define INTERNAL_CHECK_GET_ARGS_8(fmt, v1, v2, v3, v4, v5, v6, v7) , v1, v2, v3, v4, v5, v6, v7
#define INTERNAL_CHECK_GET_ARGS_9(fmt, v1, v2, v3, v4, v5, v6, v7, v8) , v1, v2, v3, v4, v5, v6, v7, v8
#define INTERNAL_CHECK_GET_ARGS_10(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9) , v1, v2, v3, v4, v5, v6, v7, v8, v9
#define INTERNAL_CHECK_GET_ARGS_11(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)                                       \
    , v1, v2, v3, v4, v5, v6, v7, v8, v9, v10
#define INTERNAL_CHECK_GET_ARGS_12(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)                                  \
    , v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11
#define INTERNAL_CHECK_GET_ARGS_13(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12)                             \
    , v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12
#define INTERNAL_CHECK_GET_ARGS_14(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13)                        \
    , v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13
#define INTERNAL_CHECK_GET_ARGS_15(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14)                   \
    , v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14
#define INTERNAL_CHECK_GET_ARGS_16(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)              \
    , v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15
