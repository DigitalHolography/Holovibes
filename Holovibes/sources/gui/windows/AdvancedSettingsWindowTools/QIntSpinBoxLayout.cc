#include "QIntSpinBoxLayout.hh"

namespace holovibes::gui
{

#define DEFAULT_MINIMUM_VALUE 0
#define DEFAULT_MAXIMUM_VALUE INT_MAX

QIntSpinBoxLayout::QIntSpinBoxLayout(QWidget* parent_widget, const std::string& name)
    : QSpinBoxLayout(parent_widget, name)
{
    spin_box_ = new QSpinBox(parent_widget);

    // spin box's default settings
    set_minimum_value(DEFAULT_MINIMUM_VALUE)->set_maximum_value(DEFAULT_MAXIMUM_VALUE);

    connect(spin_box_, SIGNAL(valueChanged(int)), this, SIGNAL(value_changed()));
    addWidget(spin_box_, Qt::AlignRight);
}

QIntSpinBoxLayout::~QIntSpinBoxLayout() { LOG_INFO; };

#pragma region SETTERS

QIntSpinBoxLayout* QIntSpinBoxLayout::set_value(int default_value)
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