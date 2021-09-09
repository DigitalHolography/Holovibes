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