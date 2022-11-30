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