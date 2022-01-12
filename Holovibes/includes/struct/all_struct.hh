#pragma once

#include <nlohmann/json_fwd.hpp>
using json = ::nlohmann::json;

#define SERIALIZE_JSON_FWD(Type)                                                                                       \
    void to_json(json& json, const Type& obj);                                                                         \
    void from_json(const json& json, Type& obj);                                                                       \
    std::ostream& operator<<(std::ostream& os, const Type& obj);