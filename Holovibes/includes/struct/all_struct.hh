#pragma once

/*!
 * \file all_struct.hh
 * \brief Defines two wrapper macros around nlohmann::json to serialize structs and enums.
 * And a syntactic sugar maccro to insert Load, Update and Assert settings related functions.
 * 
 * These macros will create two functions:
 * - `to_json`: will serialize the struct/enum to a json object.
 * - `from_json`: will deserialize the struct/enum from a json object.
 * 
 * Usage:
 * - To serialize a struct, use `SERIALIZE_JSON_STRUCT` macro with
 *   the name of the struct in the first argument and
 *   the fields/values to serialize in variadic arguments. Ex:
 * ```cpp
 * struct MyStruct
 * {
 *     int a;
 *     std::string b;
 *     std::vector<int> c;
 * 
 *     SERIALIZE_JSON_STRUCT(MyStruct, a, b, c)
 * };
 * ```
 * - To serialize an enum, use the `SERIALIZE_JSON_ENUM` macro with
 *   the name of the enum in the first argument and a map of
 *   enum value as key and the json string you want as value in the second argument. Ex:
 * ```cpp	
 * enum class Device
 * {
 *     GPU = 0,
 *     CPU
 * };
 * SERIALIZE_JSON_ENUM(Device, {
 *     {Device::GPU, "GPU"},
 *     {Device::CPU, "CPU"},
 * })
 * ```
 */

#include <nlohmann/json.hpp>

using json = ::nlohmann::json;

#define SERIALIZE_JSON_STRUCT(Type, ...) NLOHMANN_DEFINE_TYPE_INTRUSIVE(Type, __VA_ARGS__)

#define SERIALIZE_JSON_ENUM(Type, ...) NLOHMANN_JSON_SERIALIZE_ENUM(Type, __VA_ARGS__)

#define SETTING_RELATED_FUNCTIONS() \
/*! \brief Synchornize instance of ComputeSettings with GSH */ \
void Load();                        \
void Update();                      \
void Assert() const;