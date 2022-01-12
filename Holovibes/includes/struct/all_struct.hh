#pragma once

#include <string>

#include <nlohmann/json_fwd.hpp>

#define NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Type)                                                                   \
    inline void to_json(nlohmann::json& nlohmann_json_j, const Type& nlohmann_json_t);                                 \
    inline void from_json(const nlohmann::json& nlohmann_json_j, Type& nlohmann_json_t);

namespace holovibes
{
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const T& obj)
{
    return os << json{obj};
}
} // namespace holovibes