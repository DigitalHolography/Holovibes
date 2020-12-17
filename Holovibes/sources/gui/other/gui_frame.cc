/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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