#pragma once

#include <nlohmann/json.hpp>
using json = ::nlohmann::json;

#define SERIALIZE_JSON_STRUCT(Type, ...) NLOHMANN_DEFINE_TYPE_INTRUSIVE(Type, __VA_ARGS__)

#define SERIALIZE_JSON_ENUM(Type, ...) NLOHMANN_JSON_SERIALIZE_ENUM(Type, __VA_ARGS__)


// Compute settings

struct ComputeSettings
{
    Rendering image_rendering;
    Views view;
    Composite composite;
    AdvancedSettings advanced;

    SERIALIZE_JSON_STRUCT(ComputeSettings, image_rendering, view, composite, advanced)
};