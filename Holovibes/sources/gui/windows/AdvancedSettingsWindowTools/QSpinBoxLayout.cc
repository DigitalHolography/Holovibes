#include "QSpinBoxLayout.hh"

namespace holovibes::gui
{
QSpinBoxLayout::QSpinBoxLayout(QMainWindow* parent, QWidget* parent_widget, const std::string& name)
    : QHBoxLayout(parent)
{
    label_ = new QLabel(parent_widget);
    setLabel(name);
    this->addWidget(label_);

    // spin_box_ = new QSpinBox(parent_widget);
    // this->addWidget(spin_box_);
}

QSpinBoxLayout::~QSpinBoxLayout(){};

void QSpinBoxLayout::setLabel(const std::string& name) { label_->setText(QString::fromUtf8(name.c_str())); }
/*
QSpinBoxLayout* QSpinBoxLayout::setValue(int default_value)
{
    spin_box_->setValue(default_value);
    return this;
}
*/
} // namespace holovibes::gui