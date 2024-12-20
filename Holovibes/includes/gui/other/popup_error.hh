/*! \file
 *
 * \brief Declaration of the popup_error functions
 */
#pragma once

namespace holovibes::gui
{

/*! \brief Display error on a popup and exit the program after the popup is closed
 *
 * \param error_msg Message to display on the popup
 * \param exit_value Exit value of the program
 */
void show_error_and_exit(const std::string& error_msg, const int exit_value = 1);

/*! \brief Display warning on a popup
 *
 * \param warn_msg Message to display on the popup
 */
void show_warn(const std::string& warn_msg);
} // namespace holovibes::gui
