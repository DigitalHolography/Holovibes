/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

namespace holovibes::gui
{

/*! \brief Display error on a popup and exit the program after the popup is
 * closed \param error_msg Message to display on the popup \param exit_value
 * Exit value of the program
 */
void show_error_and_exit(const std::string& error_msg,
                         const int exit_value = 1);
} // namespace holovibes::gui