/*! \file
 *
 *  \brief This file is inspired by https://github.com/swansontec/map-macro which is
 *  a repository explaining how to do recursive macro.
 *  This one has been adapted to MSVC due to the post
 * https://stackoverflow.com/questions/29457312/map-macro-in-msvc-2010 Then it has been adapted to Holovibes in order to
 * work with tuples.
 */

#pragma once
// clang-format off

// combine names
#define PLUS_TEXT_(x,y) x ## y
#define PLUS_TEXT(x, y) PLUS_TEXT_(x, y)

// receive args from VA_ARGS
#define ARG_1(_1, ...) _1
#define ARG_2(_1, _2, ...) _2
#define ARG_3(_1, _2, _3, ...) _3

#define ARG_40( _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, \
                _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
                _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, \
                _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, \
                ...) _39

// slice VA_ARGS
#define OTHER_1(_1, ...) __VA_ARGS__
#define OTHER_3(_1, _2, _3, ...) __VA_ARGS__

// hack for recursion in macro
#define EVAL0(...) __VA_ARGS__
#define EVAL1(...) EVAL0 (EVAL0 (EVAL0 (__VA_ARGS__)))
#define EVAL2(...) EVAL1 (EVAL1 (EVAL1 (__VA_ARGS__)))
#define EVAL3(...) EVAL2 (EVAL2 (EVAL2 (__VA_ARGS__)))
#define EVAL4(...) EVAL3 (EVAL3 (EVAL3 (__VA_ARGS__)))
#define EVAL(...) EVAL4 (EVAL4 (EVAL4 (__VA_ARGS__)))
// expand expressions
#define EXPAND(x) x

// "return" 2 if there are args, otherwise return 0
// for MAP it's ok that ignore first arg and no case with only one arg
// IMPORTANT! must call as MAP_SWITCH(0, __VA_ARGS__) for detection 0/1 arg case
#define MAP_SWITCH(...)\
    EXPAND(ARG_40(__VA_ARGS__, 2, 2, 2, 2, 2, 2, 2, 2, 2,\
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\
            2, 2, 2, 2, 2, 2, 2, 2, 2,\
            2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0))

// macro for recursion
#define MAP_A(...) \
    PLUS_TEXT(MAP_NEXT_, MAP_SWITCH(0, __VA_ARGS__)) (MAP_B, __VA_ARGS__)

#define MAP_B(...) \
    PLUS_TEXT(MAP_NEXT_, MAP_SWITCH(0, __VA_ARGS__)) (MAP_A, __VA_ARGS__)

// for single items
// #define MAP_CALL(fn, Value) EXPAND(fn(Value))
// for tuples
#define MAP_CALL(fn, Value) EXPAND(fn Value)

#define MAP_OUT /* empty! nasty hack for recursion */

// call destination func/macro
#define MAP_NEXT_2(...)\
    MAP_CALL(EXPAND(ARG_2(__VA_ARGS__)), EXPAND(ARG_3(__VA_ARGS__))) \
    EXPAND(ARG_1(__VA_ARGS__)) \
    MAP_OUT \
    (EXPAND(ARG_2(__VA_ARGS__)), EXPAND(OTHER_3(__VA_ARGS__)))

#define MAP_NEXT_0(...) /* end mapping */

// run foreach mapping... 1st arg must be function/macro with one input argument
#define MAP(...)    EVAL(MAP_A(__VA_ARGS__))

// clang-format on
