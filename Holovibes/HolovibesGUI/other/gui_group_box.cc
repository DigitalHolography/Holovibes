#include "gui_group_box.hh"
#include "logger.hh"

namespace holovibes::gui
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
} // namespace holovibes::gui
