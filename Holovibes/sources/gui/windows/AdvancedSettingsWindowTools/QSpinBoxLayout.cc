#include "QSpinBoxLayout.hh"

namespace holovibes::gui
{
QSpinBoxLayout::QSpinBoxLayout(QMainWindow* parent, QWidget* parent_widget, const std::string& name)
    : QHBoxLayout()
{
    label_ = new QLabel(parent_widget);
    setLabel(name);
    this->addWidget(label_);
}

QSpinBoxLayout::~QSpinBoxLayout(){};

void QSpinBoxLayout::setLabel(const std::string& name) { label_->setText(QString::fromUtf8(name.c_str())); }

} // namespace holovibes::gui