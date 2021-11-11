#include "QPathSelectorLayout.hh"

namespace holovibes::gui
{

#define DEFAULT_MARGIN 7

QPathSelectorLayout::QPathSelectorLayout(QWidget* parent_widget)
    : QHBoxLayout()
{
    label_ = new QLabel(parent_widget);
    this->addWidget(label_);

    // Single blank widget only between label and line edit
    QLabel* blank = new QLabel(parent_widget);
    blank->setMargin(DEFAULT_MARGIN);
    addWidget(blank, 0, Qt::AlignRight);

    line_edit_ = new QLineEdit(parent_widget);
    this->addWidget(line_edit_, Qt::AlignRight);

    browse_button_ = new QToolButton(parent_widget);
    browse_button_->setText("...");
    connect(browse_button_, SIGNAL(clicked(bool)), this, SLOT(change_folder()));
    this->addWidget(browse_button_);
}

QPathSelectorLayout::~QPathSelectorLayout() {}

#pragma region SETTERS

QPathSelectorLayout* QPathSelectorLayout::set_text(const std::string& text)
{
    line_edit_->setText(QString::fromUtf8(text.c_str()));
    return this;
}
QPathSelectorLayout* QPathSelectorLayout::set_name(const std::string& name)
{
    label_->setText(QString::fromUtf8(name.c_str()));
    return this;
}

#pragma endregion

#pragma region GETTERS

const std::string QPathSelectorLayout::get_text() { return line_edit_->text().toStdString(); }

#pragma endregion

#pragma region SLOTS
void QPathSelectorLayout::change_folder()
{
    QString foldername =
        QFileDialog::getExistingDirectory(nullptr,
                                          tr("Open Directory"),
                                          line_edit_->text(),
                                          QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (foldername.isEmpty())
        return;

    line_edit_->setText(foldername);
    folder_changed();
}
#pragma endregion

} // namespace holovibes::gui