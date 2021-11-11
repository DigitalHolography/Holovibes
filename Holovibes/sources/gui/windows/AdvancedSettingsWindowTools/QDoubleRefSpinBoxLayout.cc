#include "QDoubleRefSpinBoxLayout.hh"
#include "logger.hh"

namespace holovibes::gui
{
QDoubleRefSpinBoxLayout::QDoubleRefSpinBoxLayout(QMainWindow* parent, const std::string& name, double* value_ptr)
    : QDoubleSpinBoxLayout(parent, name)
    , value_ptr_(value_ptr)
{
    set_value(*value_ptr_);
    connect(this, SIGNAL(value_changed()), this, SLOT(refresh_value()));
}

QDoubleRefSpinBoxLayout::~QDoubleRefSpinBoxLayout() { LOG_INFO; }

#pragma region SETTERS

QDoubleSpinBoxLayout* QDoubleRefSpinBoxLayout::set_value(double value)
{
    if (value_ptr_ == nullptr)
        return this;

    spin_box_->setValue(value);
    *value_ptr_ = value;
    return this;
}

#pragma endregion

#pragma region SLOTS

void QDoubleRefSpinBoxLayout::refresh_value()
{
    if (value_ptr_ == nullptr)
        return;

    *value_ptr_ = get_value();
    LOG_INFO << *value_ptr_;
}

#pragma endregion

} // namespace holovibes::gui