#include "q_int_spin_box_layout.hh"

namespace holovibes::gui
{

#define DEFAULT_MINIMUM_VALUE 0
#define DEFAULT_MAXIMUM_VALUE INT_MAX

QIntSpinBoxLayout::QIntSpinBoxLayout(QMainWindow* parent, const std::string& name)
    : QSpinBoxLayout(parent, name)
{
    spin_box_ = new QSpinBox();

    // spin box's default settings
    set_minimum_value(DEFAULT_MINIMUM_VALUE)->set_maximum_value(DEFAULT_MAXIMUM_VALUE);

    addWidget(spin_box_, Qt::AlignRight);
}

QIntSpinBoxLayout::~QIntSpinBoxLayout() {};

#pragma region SETTERS

QIntSpinBoxLayout* QIntSpinBoxLayout::set_value(int value)
{
    spin_box_->setValue(value);
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

QIntSpinBoxLayout* QIntSpinBoxLayout::set_single_step(int step)
{
    spin_box_->setSingleStep(step);
    return this;
}

#pragma endregion

#pragma region GETTERS

int QIntSpinBoxLayout::get_value() { return spin_box_->value(); }

#pragma endregion
} // namespace holovibes::gui