#include "ui_advancedsettingswindow.h"
#include "AdvancedSettingsWindow.hh"
#include "API.hh"
#include "user_interface_descriptor.hh"
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
    auto& api = API;

    api.record.set_record_queue_location(ui.RecordQueueLocationCheckBox->isChecked() ? Device::GPU : Device::CPU);

    api.input.set_file_buffer_size(static_cast<int>(ui.FileBSSpinBox->value()));
    api.input.set_input_buffer_size(static_cast<int>(ui.InputBSSpinBox->value()));
    api.record.set_record_buffer_size(static_cast<int>(ui.RecordBSSpinBox->value()));
    api.compute.set_output_buffer_size(static_cast<int>(ui.OutputBSSpinBox->value()));
    api.transform.set_time_transformation_cuts_output_buffer_size(static_cast<int>(ui.Cuts3DBSSpinBox->value()));

    api.view.set_display_rate(ui.DisplayRateSpinBox->value());
    api.filter2d.set_filter2d_smooth_low(ui.Filter2DLowSpinBox->value());
    api.filter2d.set_filter2d_smooth_high(ui.Filter2DHighSpinBox->value());
    api.contrast.set_contrast_lower_threshold(ui.ContrastLowerSpinBox->value());
    api.contrast.set_contrast_upper_threshold(ui.ContrastUpperSpinBox->value());
    api.global_pp.set_renorm_constant(ui.RenormConstantSpinBox->value());
    api.contrast.set_cuts_contrast_p_offset(ui.CutsContrastSpinBox->value());

    UserInterfaceDescriptor::instance().output_filename_ = ui.OutputNameLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().record_output_directory_ = ui.InputFolderPathLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().file_input_directory_ = ui.OutputFolderPathLineEdit->text().toStdString();

    UserInterfaceDescriptor::instance().auto_scale_point_threshold_ = ui.autoScalePointThresholdSpinBox->value();

    api.window_pp.set_raw_bitshift(ui.rawBitShiftSpinBox->value());

    if (callback_)
        callback_();

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
    auto& api = API;

    ui.FileBSSpinBox->setValue(api.input.get_file_buffer_size());
    ui.InputBSSpinBox->setValue(api.input.get_input_buffer_size());
    ui.RecordBSSpinBox->setValue(api.record.get_record_buffer_size());
    ui.OutputBSSpinBox->setValue(static_cast<int>(api.compute.get_output_buffer_size()));
    ui.Cuts3DBSSpinBox->setValue(api.transform.get_time_transformation_cuts_output_buffer_size());

    ui.DisplayRateSpinBox->setValue(api.view.get_display_rate());
    ui.Filter2DLowSpinBox->setValue(api.filter2d.get_filter2d_smooth_low());
    ui.Filter2DHighSpinBox->setValue(api.filter2d.get_filter2d_smooth_high());
    ui.ContrastLowerSpinBox->setValue(api.contrast.get_contrast_lower_threshold());
    ui.ContrastUpperSpinBox->setValue(api.contrast.get_contrast_upper_threshold());
    ui.RenormConstantSpinBox->setValue(api.global_pp.get_renorm_constant());
    ui.CutsContrastSpinBox->setValue(api.contrast.get_cuts_contrast_p_offset());

    ui.RecordQueueLocationCheckBox->setChecked(api.record.get_record_queue_location() == Device::GPU);

    ui.OutputNameLineEdit->setText(UserInterfaceDescriptor::instance().output_filename_.c_str());
    ui.InputFolderPathLineEdit->setText(UserInterfaceDescriptor::instance().record_output_directory_.c_str());
    ui.OutputFolderPathLineEdit->setText(UserInterfaceDescriptor::instance().file_input_directory_.c_str());

    ui.autoScalePointThresholdSpinBox->setValue(
        static_cast<int>(UserInterfaceDescriptor::instance().auto_scale_point_threshold_));

    ui.rawBitShiftSpinBox->setValue(api.window_pp.get_raw_bitshift());
}

} // namespace holovibes::gui
