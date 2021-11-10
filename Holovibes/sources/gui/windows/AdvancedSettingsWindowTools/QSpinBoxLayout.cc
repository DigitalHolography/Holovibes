#include "QSpinBoxLayout.hh"

namespace holovibes::gui
{

#define DEFAULT_SPACING 30

QSpinBoxLayout::QSpinBoxLayout(QWidget* parent_widget, const std::string& name)
    : QHBoxLayout()
{
    setSpacing(DEFAULT_SPACING);

    label_ = new QLabel(parent_widget);
    setLabel(name);
    this->addWidget(label_);
}

QSpinBoxLayout::~QSpinBoxLayout(){};

void QSpinBoxLayout::setLabel(const std::string& name) { label_->setText(QString::fromUtf8(name.c_str())); }

} // namespace holovibes::gui