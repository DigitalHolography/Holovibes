#include "QIntSpinBoxLayout.hh"

namespace holovibes::gui
{
QIntSpinBoxLayout::QIntSpinBoxLayout(QMainWindow* parent, QWidget* parent_widget, const std::string& name)
    : QSpinBoxLayout(parent, parent_widget, name)
{
    spin_box_ = new QSpinBox(parent_widget);

    // spin box's setters
    set_minimum_value(MININT)->set_maximum_value(MAXINT);

    connect(spin_box_, SIGNAL(valueChanged(int)), this, SIGNAL(value_changed()));
    addWidget(spin_box_);
}

QIntSpinBoxLayout::~QIntSpinBoxLayout(){};

#pragma region SETTERS

QIntSpinBoxLayout* QIntSpinBoxLayout::setValue(int default_value)
{
    spin_box_->setValue(default_value);
    return this;
}

QIntSpinBoxLayout* QIntSpinBoxLayout::set_minimum_value(int minimum_value)
{
    spin_box_->setMinimum(minimum_value);
    return this;
}

QIntSpinBoxLayout* QIntSpinBoxLayout::set_maximum_value(int maximum_value)
{
    spin_box_->setMaximum(maximum_value);
    return this;
}

#pragma endregion

#pragma region GETTERS

int QIntSpinBoxLayout::get_value() { return spin_box_->value(); }

#pragma endregion
} // namespace holovibes::gui