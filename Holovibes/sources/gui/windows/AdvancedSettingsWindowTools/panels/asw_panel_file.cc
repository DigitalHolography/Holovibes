#include "asw_panel_file.hh"

namespace holovibes::gui
{

ASWPanelFile::ASWPanelFile(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "File")
{
    file_layout_ = new QVBoxLayout();

    // Default input folder path selector
    default_input_folder_ = new QPathSelectorLayout(parent, parent_widget);
    default_input_folder_->setName("Default Input folder")->setText("file1");
    file_layout_->addItem(default_input_folder_);
    connect(default_input_folder_, SIGNAL(folder_changed()), this, SLOT(on_change_input_folder()));

    // Default output folder path selector
    default_output_folder_ = new QPathSelectorLayout(parent, parent_widget);
    default_output_folder_->setName("Default Output folder")->setText("file2");
    file_layout_->addItem(default_output_folder_);
    connect(default_output_folder_, SIGNAL(folder_changed()), this, SLOT(on_change_output_folder()));

    // Batch input folder path selector
    batch_input_folder_ = new QPathSelectorLayout(parent, parent_widget);
    batch_input_folder_->setName("Batch Input folder")->setText("file3");
    file_layout_->addItem(batch_input_folder_);
    connect(batch_input_folder_, SIGNAL(folder_changed()), this, SLOT(on_change_batch_input_folder()));

    setLayout(file_layout_);
}

ASWPanelFile::~ASWPanelFile() {}

#pragma region SLOTS
// TODO: region to implement with API
void ASWPanelFile::on_change_input_folder() { LOG_INFO << default_input_folder_->get_text(); }

void ASWPanelFile::on_change_output_folder() { LOG_INFO << default_output_folder_->get_text(); }

void ASWPanelFile::on_change_batch_input_folder() { LOG_INFO << batch_input_folder_->get_text(); }
#pragma endregion
} // namespace holovibes::gui