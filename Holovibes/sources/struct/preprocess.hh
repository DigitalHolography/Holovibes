#define EXPAND(x) x
#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, NAME, ...) NAME
#define PASTE(...)                                                                                                     \
    EXPAND(GET_MACRO(__VA_ARGS__,                                                                                      \
                     CONSTRUCTOR12,                                                                                    \
                     CONSTRUCTOR11,                                                                                    \
                     CONSTRUCTOR10,                                                                                    \
                     CONSTRUCTOR9,                                                                                     \
                     CONSTRUCTOR8,                                                                                     \
                     CONSTRUCTOR7,                                                                                     \
                     CONSTRUCTOR6,                                                                                     \
                     CONSTRUCTOR5,                                                                                     \
                     CONSTRUCTOR4,                                                                                     \
                     CONSTRUCTOR3,                                                                                     \
                     CONSTRUCTOR2,                                                                                     \
                     CONSTRUCTOR1)(__VA_ARGS__))

#define CONSTRUCTOR1(v1) , v1(GSH.instance().get_##v1())
#define CONSTRUCTOR2(v1, v2) CONSTRUCTOR1(v1) CONSTRUCTOR1(v2)
#define CONSTRUCTOR3(v1, v2, v3) CONSTRUCTOR1(v1) CONSTRUCTOR2(v2, v3)
#define CONSTRUCTOR4(v1, v2, v3, v4) CONSTRUCTOR1(v1) CONSTRUCTOR3(v2, v3, v4)
#define CONSTRUCTOR5(v1, v2, v3, v4, v5) CONSTRUCTOR1(v1) CONSTRUCTOR4(v2, v3, v4, v5)
#define CONSTRUCTOR6(v1, v2, v3, v4, v5, v6) CONSTRUCTOR1(v1) CONSTRUCTOR5(v2, v3, v4, v5, v6)
#define CONSTRUCTOR7(v1, v2, v3, v4, v5, v6, v7) CONSTRUCTOR1(v1) CONSTRUCTOR6(v2, v3, v4, v5, v6, v7)
#define CONSTRUCTOR8(v1, v2, v3, v4, v5, v6, v7, v8) CONSTRUCTOR1(v1) CONSTRUCTOR7(v2, v3, v4, v5, v6, v7, v8)
#define CONSTRUCTOR9(v1, v2, v3, v4, v5, v6, v7, v8, v9) CONSTRUCTOR1(v1) CONSTRUCTOR8(v2, v3, v4, v5, v6, v7, v8, v9)
#define CONSTRUCTOR10(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)                                                         \
    CONSTRUCTOR1(v1) CONSTRUCTOR9(v2, v3, v4, v5, v6, v7, v8, v9, v10)
#define CONSTRUCTOR11(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)                                                    \
    CONSTRUCTOR1(v1) CONSTRUCTOR10(v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)
#define CONSTRUCTOR12(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12)                                               \
    CONSTRUCTOR1(v1) CONSTRUCTOR11(v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12)

#define ATTRIBUTE(v1, reste...) : v1(GSH.instance().get_##v1()) PASTE(reste) {}

#define BEGIN(name, reste...) name::name(const name& name) ATTRIBUTE(reste)