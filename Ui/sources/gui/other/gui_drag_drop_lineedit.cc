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
