#include "q_double_spin_box_layout.hh"

namespace holovibes::gui
{

#define DEFAULT_MINIMUM_VALUE 0
#define DEFAULT_MAXIMUM_VALUE DBL_MAX

QDoubleSpinBoxLayout::QDoubleSpinBoxLayout(QMainWindow* parent, const std::string& name)
    : QSpinBoxLayout(parent, name)
{
    spin_box_ = new QDoubleSpinBox();

    // spin box's default settings
    set_minimum_value(DEFAULT_MINIMUM_VALUE)->set_maximum_value(DEFAULT_MAXIMUM_VALUE);

    addWidget(spin_box_, Qt::AlignRight);
}

QDoubleSpinBoxLayout::~QDoubleSpinBoxLayout(){};

#pragma region SETTERS

QDoubleSpinBoxLayout* QDoubleSpinBoxLayout::set_value(double value)
{
    spin_box_->setValue(value);
    return this;
}

QDoubleSpinBoxLayout* QDoubleSpinBoxLayout::set_minimum_value(double minimum_value)
{
    spin_box_->setMinimum(minimum_value);
    return this;
}

QDoubleSpinBoxLayout* QDoubleSpinBoxLayout::set_maximum_value(double maximum_value)
{
    spin_box_->setMaximum(maximum_value);
    return this;
}

QDoubleSpinBoxLayout* QDoubleSpinBoxLayout::set_single_step(double step)
{
    spin_box_->setSingleStep(step);
    return this;
}

QDoubleSpinBoxLayout* QDoubleSpinBoxLayout::set_decimals(int precision)
{
    spin_box_->setDecimals(precision);
    return this;
}

#pragma endregion

#pragma region GETTERS

double QDoubleSpinBoxLayout::get_value() { return spin_box_->value(); }

#pragma endregion
} // namespace holovibes::gui