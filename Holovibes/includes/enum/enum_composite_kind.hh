/*! \file
 *
 * \brief Enum for kind of composite
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum CompositeKindEnum
 *
 * \brief Represents the kind of composite image
 */
enum class CompositeKindEnum
{
    RGB = 0, /*!< Composite in RGB */
    HSV      /*!< Composite in HSV */
};

// clang-format off
SERIALIZE_JSON_ENUM(CompositeKindEnum, {
    {CompositeKindEnum::RGB, "RGB"},
    {CompositeKindEnum::HSV, "HSV"},
})
inline std::string composite_kind_to_string(const CompositeKindEnum in)
{
    return _internal::composite_kind_to_string.at(in);
}

inline CompositeKindEnum composite_kind_from_string(const std::string& in)
{
    return _internal::string_to_composite_kind.at(in);
}

inline std::ostream& operator<<(std::ostream& os, holovibes::CompositeKindEnum value)
{
    return os << _internal::composite_kind_to_string.at(value);
}

} // namespace holovibes