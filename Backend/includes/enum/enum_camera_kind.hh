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
    NONE = 0,                      /*!< No camera */
    Adimec,                        /*!< Adimec camera */
    IDS,                           /*!< IDS camera */
    Phantom,                       /*!< Phantom S710 camera */
    BitflowCyton,                  /*!< Generic bitflow cyton frame grabber */
    Hamamatsu,                     /*!< Hamamatsu camera */
    xiQ,                           /*!< xiQ camera */
    xiB,                           /*!< xiB camera */
    OpenCV,                        /*!< OpenCV camera */
    AmetekS991EuresysCoaxlinkQSFP, /*!< Ametek S991 Euresys Coaxlink QSFP+ */
    AmetekS711EuresysCoaxlinkQSFP, /*!< Ametek S711 Euresys Coaxlink QSFP+ */
    Ametek,                        /*!< Ametek camera with EGrabber Studio */
    Alvium,                        /*!< Alvium-1800-u/2050 */
    AutoDetectionPhantom,          /*!< Auto detection of Euresys' cameras */
    ASI,                           /*!< ASI camera */
};
} // namespace holovibes
