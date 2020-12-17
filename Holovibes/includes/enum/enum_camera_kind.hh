/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *  Enum for kind of camera
 */
#pragma once

namespace holovibes
{
//! \brief	Difference kind of camera supported by Holovibes
enum class CameraKind
{
    NONE = 0,  /*!< No camera */
    Adimec,    /*!< Adimec camera */
    IDS,       /*!< IDS camera */
    Phantom,   /*!< Phantom S710 camera */
    Hamamatsu, /*!< Hamamatsu camera */
    xiQ,       /*!< xiQ camera */
    xiB        /*!< xiB camera */
};
} // namespace holovibes