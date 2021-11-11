#include "QSpinBoxLayout.hh"

namespace holovibes::gui
{

#define DEFAULT_SPACING 30

QSpinBoxLayout::QSpinBoxLayout(QMainWindow* parent, const std::string& name)
    : QHBoxLayout(parent)
{
    setSpacing(DEFAULT_SPACING);

    label_ = new QLabel();
    setLabel(name);
    this->addWidget(label_);
}

QSpinBoxLayout::~QSpinBoxLayout(){};

void QSpinBoxLayout::setLabel(const std::string& name) { label_->setText(QString::fromUtf8(name.c_str())); }

} // namespace holovibes::gui