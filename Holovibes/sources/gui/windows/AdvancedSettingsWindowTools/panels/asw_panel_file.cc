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

    // Default output folder path selector
    default_output_folder_ = new QPathSelectorLayout(parent, parent_widget);
    default_output_folder_->setName("Default Output folder")->setText("file2");
    file_layout_->addItem(default_output_folder_);

    // Batch input folder path selector
    batch_input_folder_ = new QPathSelectorLayout(parent, parent_widget);
    batch_input_folder_->setName("Batch Input folder")->setText("file3");
    file_layout_->addItem(batch_input_folder_);

    setLayout(file_layout_);
}

ASWPanelFile::~ASWPanelFile() {}

} // namespace holovibes::gui