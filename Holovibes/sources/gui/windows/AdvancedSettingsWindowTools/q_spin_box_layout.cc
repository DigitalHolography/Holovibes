#include "q_spin_box_layout.hh"

namespace holovibes::gui
{

#define DEFAULT_SPACING 30

QSpinBoxLayout::QSpinBoxLayout(QMainWindow* parent, const std::string& name)
    : QHBoxLayout(parent)
{
    setSpacing(DEFAULT_SPACING);

    label_ = new QLabel();
    set_label(name);
    this->addWidget(label_);
}

QSpinBoxLayout::~QSpinBoxLayout(){};

void QSpinBoxLayout::set_label(const std::string& name) { label_->setText(QString::fromUtf8(name.c_str())); }

} // namespace holovibes::gui