#include "QIntSpinBoxLayout.hh"

namespace holovibes::gui
{
QIntSpinBoxLayout::QIntSpinBoxLayout(QMainWindow* parent, QWidget* parent_widget, const std::string& name)
    : QSpinBoxLayout(parent, parent_widget, name)
{
    spin_box_ = new QSpinBox(parent_widget);
    addWidget(spin_box_);
}

QIntSpinBoxLayout::~QIntSpinBoxLayout(){};

QIntSpinBoxLayout* QIntSpinBoxLayout::setValue(int default_value)
{
    spin_box_->setValue(default_value);
    return this;
}
} // namespace holovibes::gui