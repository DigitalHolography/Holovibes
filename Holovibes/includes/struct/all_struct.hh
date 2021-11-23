#pragma once

#include <string>

#include <nlohmann/json.hpp>
using json = ::nlohmann::json;

namespace holovibes
{
struct json_struct
{
    virtual json to_json() const = 0;
};
} // namespace holovibes