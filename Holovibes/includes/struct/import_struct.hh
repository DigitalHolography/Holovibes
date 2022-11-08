/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "types.hh"
#include "all_struct.hh"

namespace holovibes
{

/*! \enum CameraKind
 *
 * \brief Difference kind of camera supported by Holovibes
 */
enum class CameraKind
{
    None = 0,     /*!< No camera */
    Adimec,       /*!< Adimec camera */
    IDS,          /*!< IDS camera */
    Phantom,      /*!< Phantom S710 camera */
    BitflowCyton, /*!< Generic bitflow cyton frame grabber */
    Hamamatsu,    /*!< Hamamatsu camera */
    xiQ,          /*!< xiQ camera */
    xiB,          /*!< xiB camera */
    OpenCV        /*!< OpenCV camera */
};

enum ImportTypeEnum
{
    None,
    Camera,
    File,
};

// clang-format off
SERIALIZE_JSON_ENUM(CameraKind, {
    {CameraKind::None, "None"},
    {CameraKind::Adimec, "Adimec"},
    {CameraKind::IDS, "IDS"},
    {CameraKind::Phantom, "Phantom"},
    {CameraKind::BitflowCyton, "BitflowCyton"},
    {CameraKind::Hamamatsu, "Hamamatsu"},
    {CameraKind::xiQ, "xiQ"},
    {CameraKind::xiB, "xiB"},
    {CameraKind::OpenCV, "OpenCV"}
})

SERIALIZE_JSON_ENUM(ImportTypeEnum, {
    {ImportTypeEnum::None, "None"},
    {ImportTypeEnum::Camera, "Camera"},
    {ImportTypeEnum::File, "File"},
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const CameraKind& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ImportTypeEnum& value) { return os << json{value}; }

} // namespace holovibes
