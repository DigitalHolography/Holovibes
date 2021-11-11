#include "QDoubleSpinBoxLayout.hh"

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

    connect(spin_box_, SIGNAL(valueChanged(double)), this, SIGNAL(value_changed()));
    addWidget(spin_box_, Qt::AlignRight);
}

QDoubleSpinBoxLayout::~QDoubleSpinBoxLayout() { LOG_INFO; };

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

#pragma endregion

#pragma region GETTERS

double QDoubleSpinBoxLayout::get_value() { return spin_box_->value(); }

#pragma endregion
} // namespace holovibes::gui