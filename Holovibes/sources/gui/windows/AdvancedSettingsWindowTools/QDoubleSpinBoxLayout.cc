#include "QDoubleSpinBoxLayout.hh"

namespace holovibes::gui
{
QDoubleSpinBoxLayout::QDoubleSpinBoxLayout(QMainWindow* parent, QWidget* parent_widget, const std::string& name)
    : QSpinBoxLayout(parent, parent_widget, name)
{
    spin_box_ = new QDoubleSpinBox(parent_widget);
    addWidget(spin_box_);
}

QDoubleSpinBoxLayout::~QDoubleSpinBoxLayout(){};

QDoubleSpinBoxLayout* QDoubleSpinBoxLayout::setValue(double value)
{
    spin_box_->setValue(value);
    return this;
}
} // namespace holovibes::gui