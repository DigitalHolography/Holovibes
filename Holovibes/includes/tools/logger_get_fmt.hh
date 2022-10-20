/*! \file
 * macros to get the first argument
 */

/*
 * TODO: the day __VA_OPT__ is usable, this file can be replaced by 2 lines:
 *
 * #define INTERNAL_CHECK_GET_FMT_(fmt, ...) fmt
 * #define INTERNAL_CHECK_GET_FMT(...) __VA_OPT__(INTERNAL_CHECK_GET_FMT_(__VA_ARGS__))
 */

#define INTERNAL_CHECK_GET_FMT_EXPAND(x) x
#define INTERNAL_CHECK_GET_FMT_GET_MACRO(_1,                                                                           \
                                         _2,                                                                           \
                                         _3,                                                                           \
                                         _4,                                                                           \
                                         _5,                                                                           \
                                         _6,                                                                           \
                                         _7,                                                                           \
                                         _8,                                                                           \
                                         _9,                                                                           \
                                         _10,                                                                          \
                                         _11,                                                                          \
                                         _12,                                                                          \
                                         _13,                                                                          \
                                         _14,                                                                          \
                                         _15,                                                                          \
                                         _16,                                                                          \
                                         NAME,                                                                         \
                                         ...)                                                                          \
    NAME

/*! \brief
 * macro to be used
 * at most 16 arguments
 */
#define INTERNAL_CHECK_GET_FMT(...)                                                                                    \
    INTERNAL_CHECK_GET_FMT_EXPAND(INTERNAL_CHECK_GET_FMT_GET_MACRO(__VA_ARGS__,                                        \
                                                                   INTERNAL_CHECK_GET_FMT_16,                          \
                                                                   INTERNAL_CHECK_GET_FMT_15,                          \
                                                                   INTERNAL_CHECK_GET_FMT_14,                          \
                                                                   INTERNAL_CHECK_GET_FMT_13,                          \
                                                                   INTERNAL_CHECK_GET_FMT_12,                          \
                                                                   INTERNAL_CHECK_GET_FMT_11,                          \
                                                                   INTERNAL_CHECK_GET_FMT_10,                          \
                                                                   INTERNAL_CHECK_GET_FMT_9,                           \
                                                                   INTERNAL_CHECK_GET_FMT_8,                           \
                                                                   INTERNAL_CHECK_GET_FMT_7,                           \
                                                                   INTERNAL_CHECK_GET_FMT_6,                           \
                                                                   INTERNAL_CHECK_GET_FMT_5,                           \
                                                                   INTERNAL_CHECK_GET_FMT_4,                           \
                                                                   INTERNAL_CHECK_GET_FMT_3,                           \
                                                                   INTERNAL_CHECK_GET_FMT_2,                           \
                                                                   INTERNAL_CHECK_GET_FMT_1)(__VA_ARGS__))
#define INTERNAL_CHECK_GET_FMT_1(fmt) fmt
#define INTERNAL_CHECK_GET_FMT_2(fmt, v1) fmt
#define INTERNAL_CHECK_GET_FMT_3(fmt, v1, v2) fmt
#define INTERNAL_CHECK_GET_FMT_4(fmt, v1, v2, v3) fmt
#define INTERNAL_CHECK_GET_FMT_5(fmt, v1, v2, v3, v4) fmt
#define INTERNAL_CHECK_GET_FMT_6(fmt, v1, v2, v3, v4, v5) fmt
#define INTERNAL_CHECK_GET_FMT_7(fmt, v1, v2, v3, v4, v5, v6) fmt
#define INTERNAL_CHECK_GET_FMT_8(fmt, v1, v2, v3, v4, v5, v6, v7) fmt
#define INTERNAL_CHECK_GET_FMT_9(fmt, v1, v2, v3, v4, v5, v6, v7, v8) fmt
#define INTERNAL_CHECK_GET_FMT_10(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9) fmt
#define INTERNAL_CHECK_GET_FMT_11(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10) fmt
#define INTERNAL_CHECK_GET_FMT_12(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) fmt
#define INTERNAL_CHECK_GET_FMT_13(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12) fmt
#define INTERNAL_CHECK_GET_FMT_14(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13) fmt
#define INTERNAL_CHECK_GET_FMT_15(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14) fmt
#define INTERNAL_CHECK_GET_FMT_16(fmt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) fmt