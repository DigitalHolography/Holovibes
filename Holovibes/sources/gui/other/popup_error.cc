/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "popup_error.hh"
#include "holovibes.hh"

namespace holovibes::gui
{
void show_error_and_exit(const std::string& error_msg, const int exit_value)
{
    QMessageBox messageBox;
    messageBox.critical(nullptr, "Internal Error", error_msg.c_str());
    exit(exit_value);
}
} // namespace holovibes::gui