/*! \file
 *
 */

#include "panel.hh"

#include "MainWindow.hh"

namespace holovibes::gui
{
Panel::Panel(QWidget* parent)
    : QGroupBox(parent)
    , parent_(find_main_window(parent))
    , ui_(parent_->get_ui())
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

void Panel::QSpinBoxQuietSetValue(QSpinBox* spinBox, int value)
{
    spinBox->blockSignals(true);
    spinBox->setValue(value);
    spinBox->blockSignals(false);
}

void Panel::QSliderQuietSetValue(QSlider* slider, int value)
{
    slider->blockSignals(true);
    slider->setValue(value);
    slider->blockSignals(false);
}

void Panel::QDoubleSpinBoxQuietSetValue(QDoubleSpinBox* spinBox, double value)
{
    spinBox->blockSignals(true);
    spinBox->setValue(value);
    spinBox->blockSignals(false);
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
