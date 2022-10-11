/*! \file
 *
 * \brief Enum for kind of composite
 */
#pragma once

#include <map>

namespace holovibes
{
/*! \enum CompositeKind
 *
 * \brief Represents the kind of composite image
 */
enum class CompositeKind
{
    RGB = 0, /*!< Composite in RGB */
    HSV      /*!< Composite in HSV */
};
} // namespace holovibes

namespace holovibes::_internal
{
const std::map<CompositeKind, std::string> composite_kind_to_string = {
    {CompositeKind::RGB, "RGB"},
    {CompositeKind::HSV, "HSV"},
};

const std::map<std::string, CompositeKind> string_to_composite_kind = {
    {"RGB", CompositeKind::RGB},
    {"HSV", CompositeKind::HSV},
};

} // namespace holovibes::_internal

namespace holovibes
{
inline std::string composite_kind_to_string(const CompositeKind in)
{
    return _internal::composite_kind_to_string.at(in);
}

inline CompositeKind composite_kind_from_string(const std::string& in)
{
    return _internal::string_to_composite_kind.at(in);
}

inline std::ostream& operator<<(std::ostream& os, holovibes::CompositeKind value)
{
    return os << _internal::composite_kind_to_string.at(value);
}

} // namespace holovibes
