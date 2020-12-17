/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *  Enum for kind of window
 */
#pragma once

namespace holovibes
{
/*! \brief Represents the kind of slice displayed by the window */
enum class WindowKind
{
    XYview = 0, /*!< Main view */
    XZview,     /*!< view slice */
    YZview      /*!< YZ view slice */
};
} // namespace holovibes