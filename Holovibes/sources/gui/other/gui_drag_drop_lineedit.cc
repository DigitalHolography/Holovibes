/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "stdafx.hh"
#include "gui_drag_drop_lineedit.hh"

namespace holovibes::gui
{

Drag_drop_lineedit::Drag_drop_lineedit(QWidget* parent)
    : QLineEdit(parent)
{
    setPlaceholderText("Drop file here");
}

void Drag_drop_lineedit::dropEvent(QDropEvent* event)
{
    auto url = event->mimeData()->urls()[0];
    auto path = url.path();
    if (path.at(0) == '/')
        path.remove(0, 1);
    setText(path);
}
} // namespace holovibes::gui
