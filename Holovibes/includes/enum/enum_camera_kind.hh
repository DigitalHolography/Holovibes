/*! \file
 *
 * \brief Enum for kind of camera
 */
#pragma once

namespace holovibes
{
/*! \enum CameraKind
 *
 * \brief Difference kind of camera supported by Holovibes
 */
enum class CameraKind
{
    NONE = 0,     /*!< No camera */
    Adimec,       /*!< Adimec camera */
    IDS,          /*!< IDS camera */
    Phantom,      /*!< Phantom S710 camera */
    BitflowCyton, /*!< Generic bitflow cyton frame grabber */
    Hamamatsu,    /*!< Hamamatsu camera */
    xiQ,          /*!< xiQ camera */
    xiB,          /*!< xiB camera */
    OpenCV,       /*!< OpenCV camera */
};
} // namespace holovibes
