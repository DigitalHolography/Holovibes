#pragma once

#include <nlohmann/json.hpp>
using json = ::nlohmann::json;

#define SERIALIZE_JSON_STRUCT(Type, ...) NLOHMANN_DEFINE_TYPE_INTRUSIVE(Type, __VA_ARGS__)

#define SERIALIZE_JSON_ENUM(Type, ...) NLOHMANN_JSON_SERIALIZE_ENUM(Type, __VA_ARGS__)