#include "gui_frame.hh"

namespace holovibes::gui
{
Frame::Frame(QWidget* parent)
    : QFrame(parent)
{
}

Frame::~Frame() {}

void Frame::ShowOrHide()
{
    if (this->isVisible())
        hide();
    else
        show();
}
} // namespace holovibes::gui
