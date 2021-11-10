#include "asw_panel_file.hh"

namespace holovibes::gui
{

#define DEFAULT_DEFAULT_INPUT_FOLDER UserInterfaceDescriptor::instance().record_output_directory_.c_str()
#define DEFAULT_DEFAULT_OUTPUT_FOLDER UserInterfaceDescriptor::instance().file_input_directory_.c_str()
#define DEFAULT_BATCH_INPUT_FOLDER UserInterfaceDescriptor::instance().batch_input_directory_.c_str()

ASWPanelFile::ASWPanelFile(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "File")
{
    file_layout_ = new QVBoxLayout();

    // Widgets creation
    create_default_input_folder_widget();
    create_default_output_folder_widget();
    create_batch_input_folder_widget();

    setLayout(file_layout_);
}

ASWPanelFile::~ASWPanelFile() {}

#pragma region WIDGETS

void ASWPanelFile::create_default_input_folder_widget()
{
    // Default input folder path selector
    default_input_folder_ = new QPathSelectorLayout(parent_widget_);
    default_input_folder_->setName("Default Input folder")->setText(DEFAULT_DEFAULT_INPUT_FOLDER);
    file_layout_->addItem(default_input_folder_);
    connect(default_input_folder_, SIGNAL(folder_changed()), this, SLOT(on_change_input_folder()));
}

void ASWPanelFile::create_default_output_folder_widget()
{
    // Default output folder path selector
    default_output_folder_ = new QPathSelectorLayout(parent_widget_);
    default_output_folder_->setName("Default Output folder")->setText(DEFAULT_DEFAULT_OUTPUT_FOLDER);
    file_layout_->addItem(default_output_folder_);
    connect(default_output_folder_, SIGNAL(folder_changed()), this, SLOT(on_change_output_folder()));
}

void ASWPanelFile::create_batch_input_folder_widget()
{
    // Batch input folder path selector
    batch_input_folder_ = new QPathSelectorLayout(parent_widget_);
    batch_input_folder_->setName("Batch Input folder")->setText(DEFAULT_BATCH_INPUT_FOLDER);
    file_layout_->addItem(batch_input_folder_);
    connect(batch_input_folder_, SIGNAL(folder_changed()), this, SLOT(on_change_batch_input_folder()));
}

#pragma endregion

#pragma region SLOTS
// TODO: region to implement with API
void ASWPanelFile::on_change_input_folder()
{
    LOG_INFO << default_input_folder_->get_text();
    UserInterfaceDescriptor::instance().record_output_directory_ = default_input_folder_->get_text();
}

void ASWPanelFile::on_change_output_folder()
{
    LOG_INFO << default_output_folder_->get_text();
    UserInterfaceDescriptor::instance().file_input_directory_ = default_output_folder_->get_text();
}

void ASWPanelFile::on_change_batch_input_folder()
{
    LOG_INFO << batch_input_folder_->get_text();
    UserInterfaceDescriptor::instance().batch_input_directory_ = batch_input_folder_->get_text();
}
#pragma endregion
} // namespace holovibes::gui