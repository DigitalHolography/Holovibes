/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "gui_group_box.hh"

namespace holovibes
{
namespace gui
{
GroupBox::GroupBox(QWidget* parent)
    : QGroupBox(parent)
{
}

GroupBox::~GroupBox() {}

void GroupBox::ShowOrHide()
{
    if (this->isVisible())
        hide();
    else
        show();
}
} // namespace gui
} // namespace holovibes