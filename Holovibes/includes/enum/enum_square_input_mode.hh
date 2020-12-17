/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *  Enum for kind of input mode
 */
#pragma once

namespace holovibes
{
//! Enum for the different input mode
enum class SquareInputMode
{
    NO_MODIFICATION = 0, /*!< No modification on the input */
    ZERO_PADDED_SQUARE, /*!< Pad the input in order to process a square input */
    CROPPED_SQUARE /*!< Crop the input in order to process a sqaure input */
};
} // namespace holovibes