/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *  Enum for kind of composite
 */
#pragma once

namespace holovibes
{
/*! \brief Represents the kind of composite image */
enum class CompositeKind
{
    RGB = 0, /*!< Composite in RGB */
    HSV      /*!< Composite in HSV */
};
} // namespace holovibes