/*! \file
 *
 * \brief Enum for the different type of import
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum ImportType
 *
 * \brief How the data is imported. Either by camera, file or none.
 */
enum ImportType
{
    None,
    Camera,
    File,
};

// clang-format off
SERIALIZE_JSON_ENUM(ImportType, {
    {ImportType::None, "NONE"},
    {ImportType::Camera, "CAMERA"},
    {ImportType::File, "FILE"},
})
// clang-format on

} // namespace holovibes
