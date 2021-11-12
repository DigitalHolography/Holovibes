#include "QLineEditLayout.hh"

namespace holovibes::gui
{

#define DEFAULT_MARGIN 7

QLineEditLayout::QLineEditLayout(QMainWindow* parent, const std::string& name)
    : QHBoxLayout(parent)
{
    label_ = new QLabel();
    set_name(name);
    addWidget(label_);

    // Single blank widget only between label and line edit
    QLabel* blank = new QLabel();
    blank->setMargin(DEFAULT_MARGIN);
    addWidget(blank, 0, Qt::AlignRight);

    line_edit_ = new QLineEdit();
    addWidget(line_edit_, Qt::AlignRight);
    connect(line_edit_, SIGNAL(textChanged(const QString&)), this, SIGNAL(text_changed()));
}

QLineEditLayout::~QLineEditLayout(){};

#pragma region SETTERS

QLineEditLayout* QLineEditLayout::set_name(const std::string& name)
{
    label_->setText(QString::fromUtf8(name.c_str()));
    return this;
}

QLineEditLayout* QLineEditLayout::set_text(const std::string& text)
{
    line_edit_->setText(QString::fromUtf8(text.c_str()));
    return this;
}

#pragma endregion

#pragma region GETTERS

const std::string QLineEditLayout::get_text() { return line_edit_->text().toStdString(); }

#pragma endregion

} // namespace holovibes::gui