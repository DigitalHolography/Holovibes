#include "panel.hh"

#include "MainWindow.hh"

namespace holovibes::gui
{
Panel::Panel(QWidget* parent)
    : QGroupBox(parent)
{
}

Panel::~Panel() {}

void Panel::ShowOrHide()
{
    if (this->isVisible())
        hide();
    else
        show();
}

MainWindow* Panel::find_main_window(QObject* widget)
{
    if (widget == nullptr)
        return nullptr;

    if (MainWindow* parent = dynamic_cast<MainWindow*>(widget))
    {
        return parent;
    }
    return find_main_window(widget->parent());
}
} // namespace holovibes::gui
