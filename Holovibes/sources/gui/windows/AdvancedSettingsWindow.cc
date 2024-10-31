#include "ui_advancedsettingswindow.h"
#include "AdvancedSettingsWindow.hh"
#include "API.hh"
#include <spdlog/spdlog.h>

namespace holovibes::gui
{

AdvancedSettingsWindow::AdvancedSettingsWindow(QMainWindow* parent)
    : QMainWindow(parent)
{

    ui.setupUi(this);
    setWindowIcon(QIcon(":/assets/icons/holovibes_logo.png"));
    this->setAttribute(Qt::WA_DeleteOnClose);
    this->show();

    set_current_values();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow()
{
    UserInterfaceDescriptor::instance().advanced_settings_window_.release();
}

#pragma region SLOTS

void AdvancedSettingsWindow::closeEvent(QCloseEvent* event) { emit closed(); }

void AdvancedSettingsWindow::set_ui_values()
{
    api::set_record_queue_location(ui.RecordQueueLocationCheckBox->isChecked() ? Device::GPU : Device::CPU);
    api::set_record_on_gpu(!ui.RecordDeviceCheckbox->isChecked());

    api::set_file_buffer_size(static_cast<int>(ui.FileBSSpinBox->value()));
    api::set_input_buffer_size(static_cast<int>(ui.InputBSSpinBox->value()));
    api::set_record_buffer_size(static_cast<int>(ui.RecordBSSpinBox->value()));
    api::set_output_buffer_size(static_cast<int>(ui.OutputBSSpinBox->value()));
    api::set_time_transformation_cuts_output_buffer_size(static_cast<int>(ui.Cuts3DBSSpinBox->value()));

    api::set_display_rate(ui.DisplayRateSpinBox->value());
    api::set_filter2d_smooth_low(ui.Filter2DLowSpinBox->value());
    api::set_filter2d_smooth_high(ui.Filter2DHighSpinBox->value());
    api::set_contrast_lower_threshold(ui.ContrastLowerSpinBox->value());
    api::set_contrast_upper_threshold(ui.ContrastUpperSpinBox->value());
    api::set_renorm_constant(ui.RenormConstantSpinBox->value());
    api::set_cuts_contrast_p_offset(ui.CutsContrastSpinBox->value());

    UserInterfaceDescriptor::instance().output_filename_ = ui.OutputNameLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().record_output_directory_ = ui.InputFolderPathLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().file_input_directory_ = ui.OutputFolderPathLineEdit->text().toStdString();

    UserInterfaceDescriptor::instance().auto_scale_point_threshold_ = ui.autoScalePointThresholdSpinBox->value();

    api::set_raw_bitshift(ui.rawBitShiftSpinBox->value());

    UserInterfaceDescriptor::instance().has_been_updated = true;
    this->close();
}

void AdvancedSettingsWindow::change_input_folder_path() { change_folder(ui.InputFolderPathLineEdit); }
void AdvancedSettingsWindow::change_output_folder_path() { change_folder(ui.OutputFolderPathLineEdit); }

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
    ui.FileBSSpinBox->setValue(api::get_file_buffer_size());
    ui.InputBSSpinBox->setValue(api::get_input_buffer_size());
    ui.RecordBSSpinBox->setValue(api::get_record_buffer_size());
    ui.OutputBSSpinBox->setValue(static_cast<int>(api::get_output_buffer_size()));
    ui.Cuts3DBSSpinBox->setValue(api::get_time_transformation_cuts_output_buffer_size());

    ui.DisplayRateSpinBox->setValue(api::get_display_rate());
    ui.Filter2DLowSpinBox->setValue(api::get_filter2d_smooth_low());
    ui.Filter2DHighSpinBox->setValue(api::get_filter2d_smooth_high());
    ui.ContrastLowerSpinBox->setValue(api::get_contrast_lower_threshold());
    ui.ContrastUpperSpinBox->setValue(api::get_contrast_upper_threshold());
    ui.RenormConstantSpinBox->setValue(api::get_renorm_constant());
    ui.CutsContrastSpinBox->setValue(api::get_cuts_contrast_p_offset());

    ui.RecordQueueLocationCheckBox->setChecked(api::get_record_queue_location() == Device::GPU);
    ui.RecordQueueLocationCheckBox->setEnabled(api::get_input_queue_location() == Device::GPU);
    ui.RecordDeviceCheckbox->setChecked(!api::get_record_on_gpu());

    ui.OutputNameLineEdit->setText(UserInterfaceDescriptor::instance().output_filename_.c_str());
    ui.InputFolderPathLineEdit->setText(UserInterfaceDescriptor::instance().record_output_directory_.c_str());
    ui.OutputFolderPathLineEdit->setText(UserInterfaceDescriptor::instance().file_input_directory_.c_str());

    ui.autoScalePointThresholdSpinBox->setValue(
        static_cast<int>(UserInterfaceDescriptor::instance().auto_scale_point_threshold_));

    ui.rawBitShiftSpinBox->setValue(api::get_raw_bitshift());
}

} // namespace holovibes::gui
