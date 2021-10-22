#include "gui_frame.hh"

namespace holovibes
{
namespace gui
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
} // namespace gui
} // namespace holovibes
