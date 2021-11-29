#pragma once

#include <string>

#include <nlohmann/json.hpp>
using json = ::nlohmann::json;

namespace holovibes
{
// TODO: Overide function in composite_struct.hh and view_struct.hh
//       Due to undefined behaviour (maybe MSVC compiler), strange things append when we use json_struct as parent.
//       Ex1: Middle computer 15-20; Cannot open some HOLO files without crashing
//       Ex2: Right computer 15-20; Cannot close Holovibes windows without segfault
//
// struct json_struct
// {
//     virtual json to_json() const = 0;
//     virtual void from_json(const json& data) = 0;
// };
} // namespace holovibes