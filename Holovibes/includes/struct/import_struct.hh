/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "types.hh"

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
    xiB           /*!< xiB camera */
};

enum ImportTypeEnum
{
    None,
    Camera,
    File,
};

inline std::ostream& operator<<(std::ostream& os, const CameraKind& value)
{
    if (value == CameraKind::None)
        return os << "None";
    if (value == CameraKind::Adimec)
        return os << "Adimec";
    if (value == CameraKind::IDS)
        return os << "IDS";
    if (value == CameraKind::Phantom)
        return os << "Phantom";
    if (value == CameraKind::BitflowCyton)
        return os << "BitflowCyton";
    if (value == CameraKind::Hamamatsu)
        return os << "Hamamatsu";
    if (value == CameraKind::xiQ)
        return os << "xiQ";
    if (value == CameraKind::xiB)
        return os << "xiB";

    return os;
}

inline std::ostream& operator<<(std::ostream& os, const ImportTypeEnum& value)
{
    if (value == ImportTypeEnum::None)
        return os << "None";
    if (value == ImportTypeEnum::Camera)
        return os << "Camera";
    if (value == ImportTypeEnum::File)
        return os << "File";

    return os;
}

} // namespace holovibes
