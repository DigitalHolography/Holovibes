#include "ui_advancedsettingswindow.h"
#include "AdvancedSettingsWindow.hh"
#include "API.hh"

namespace holovibes::gui
{

AdvancedSettingsWindow::AdvancedSettingsWindow(QMainWindow* parent, AdvancedSettingsWindowPanel* specific_panel)
    : QMainWindow(parent)
{

    ui.setupUi(this);
    setWindowIcon(QIcon(":/Holovibes.ico"));
    this->setAttribute(Qt::WA_DeleteOnClose);
    this->show();

    plug_specific_panel(specific_panel);

    set_current_values();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow()
{
    UserInterfaceDescriptor::instance().advanced_settings_window_.release();
}

#pragma region PANELS

void AdvancedSettingsWindow::plug_specific_panel(AdvancedSettingsWindowPanel* specific_panel)
{
    specific_panel_ = specific_panel;

    if (specific_panel == nullptr)
        return;

    ui.gridLayout->addWidget(specific_panel, 4, 1, 1, 1);
}

#pragma endregion

#pragma region SLOTS

void AdvancedSettingsWindow::closeEvent(QCloseEvent* event) { emit closed(); }

void AdvancedSettingsWindow::set_ui_values()
{
    api::get_cd().set_file_buffer_size(static_cast<int>(ui.FileBSSpinBox->value()));
    api::get_cd().set_input_buffer_size(static_cast<int>(ui.InputBSSpinBox->value()));
    api::get_cd().set_record_buffer_size(static_cast<int>(ui.RecordBSSpinBox->value()));
    api::get_cd().set_output_buffer_size(static_cast<int>(ui.OutputBSSpinBox->value()));
    api::get_cd().set_time_transformation_cuts_output_buffer_size(static_cast<int>(ui.Cuts3DBSSpinBox->value()));

    api::get_cd().set_display_rate(ui.DisplayRateSpinBox->value());
    api::get_cd().set_filter2d_smooth_low(ui.Filter2DLowSpinBox->value());
    api::get_cd().set_filter2d_smooth_high(ui.Filter2DHighSpinBox->value());
    api::get_cd().set_contrast_lower_threshold(ui.ContrastLowerSpinBox->value());
    api::get_cd().set_contrast_upper_threshold(ui.ContrastUpperSpinBox->value());
    api::get_cd().set_renorm_constant(ui.RenormConstantSpinBox->value());
    api::get_cd().set_cuts_contrast_p_offset(ui.CutsContrastSpinBox->value());

    UserInterfaceDescriptor::instance().default_output_filename_ = ui.OutputNameLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().record_output_directory_ = ui.InputFolderPathLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().file_input_directory_ = ui.OutputFolderPathLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().batch_input_directory_ = ui.BatchFolderPathLineEdit->text().toStdString();

    UserInterfaceDescriptor::instance().auto_scale_point_threshold_ = ui.autoScalePointThresholdSpinBox->value();

    api::get_cd().set_raw_bitshift(ui.rawBitShiftSpinBox->value());

    if (specific_panel_ != nullptr)
        specific_panel_->set_ui_values();

    UserInterfaceDescriptor::instance().has_been_updated = true;
}

void AdvancedSettingsWindow::change_input_folder_path() { change_folder(ui.InputFolderPathLineEdit); }
void AdvancedSettingsWindow::change_output_folder_path() { change_folder(ui.OutputFolderPathLineEdit); }
void AdvancedSettingsWindow::change_batch_input_folder_path() { change_folder(ui.BatchFolderPathLineEdit); }

#pragma endregion

void AdvancedSettingsWindow::change_folder(Drag_drop_lineedit* lineEdit)
{
    QString foldername =
        QFileDialog::getExistingDirectory(this,
                                          tr("Open Directory"),
                                          lineEdit->text(),
                                          QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (foldername.isEmpty())
        return;

    lineEdit->setText(foldername);
}

void AdvancedSettingsWindow::set_current_values()
{
    ui.FileBSSpinBox->setValue(api::get_cd().get_file_buffer_size());
    ui.InputBSSpinBox->setValue(api::get_cd().get_input_buffer_size());
    ui.RecordBSSpinBox->setValue(api::get_cd().get_record_buffer_size());
    ui.OutputBSSpinBox->setValue(api::get_cd().get_output_buffer_size());
    ui.Cuts3DBSSpinBox->setValue(api::get_cd().get_time_transformation_cuts_output_buffer_size());

    ui.DisplayRateSpinBox->setValue(api::get_cd().get_display_rate());
    ui.Filter2DLowSpinBox->setValue(api::get_cd().get_filter2d_smooth_low());
    ui.Filter2DHighSpinBox->setValue(api::get_cd().get_filter2d_smooth_high());
    ui.ContrastLowerSpinBox->setValue(api::get_cd().get_contrast_lower_threshold());
    ui.ContrastUpperSpinBox->setValue(api::get_cd().get_contrast_upper_threshold());
    ui.RenormConstantSpinBox->setValue(api::get_cd().get_renorm_constant());
    ui.CutsContrastSpinBox->setValue(api::get_cd().get_cuts_contrast_p_offset());

    ui.OutputNameLineEdit->setText(UserInterfaceDescriptor::instance().default_output_filename_.c_str());
    ui.InputFolderPathLineEdit->setText(UserInterfaceDescriptor::instance().record_output_directory_.c_str());
    ui.OutputFolderPathLineEdit->setText(UserInterfaceDescriptor::instance().file_input_directory_.c_str());
    ui.BatchFolderPathLineEdit->setText(UserInterfaceDescriptor::instance().batch_input_directory_.c_str());

    ui.autoScalePointThresholdSpinBox->setValue(
        static_cast<int>(UserInterfaceDescriptor::instance().auto_scale_point_threshold_));

    ui.rawBitShiftSpinBox->setValue(api::get_cd().get_raw_bitshift());

    if (specific_panel_ != nullptr)
        specific_panel_->set_current_values();
}

} // namespace holovibes::gui
