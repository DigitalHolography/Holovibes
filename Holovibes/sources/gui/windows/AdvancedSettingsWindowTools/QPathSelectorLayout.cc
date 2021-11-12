#include "QPathSelectorLayout.hh"
#include "logger.hh"
namespace holovibes::gui
{

QPathSelectorLayout::QPathSelectorLayout(QMainWindow* parent, const std::string& name)
    : QLineEditLayout(parent, name)
{
    browse_button_ = new QToolButton();
    browse_button_->setText("...");
    connect(browse_button_, SIGNAL(clicked(bool)), this, SLOT(change_folder()));
    addWidget(browse_button_);
}

QPathSelectorLayout::~QPathSelectorLayout() {}

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
}

#pragma endregion

} // namespace holovibes::gui