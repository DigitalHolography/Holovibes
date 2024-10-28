#include "popup_error.hh"
#include "holovibes.hh"

namespace holovibes::gui
{
void show_error_and_exit(const std::string& error_msg, const int exit_value)
{
    if (!holovibes::Holovibes::instance().is_cli)
    {
        QMessageBox messageBox;
        messageBox.critical(nullptr, "Internal Error", error_msg.c_str());
        exit(exit_value);
    }
    else
    {
        LOG_ERROR(error_msg);
        exit(exit_value);
    }
}

void show_warn(const std::string& warn_msg)
{
    if (!holovibes::Holovibes::instance().is_cli)
    {
        QMessageBox messageBox;
        messageBox.warning(nullptr, "Warning!", warn_msg.c_str());
    }
    else
    {
        LOG_WARN(warn_msg);
    }
}

} // namespace holovibes::gui
