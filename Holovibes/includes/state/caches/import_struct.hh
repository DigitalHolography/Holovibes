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

} // namespace holovibes
