#include "ui_advancedsettingswindow.h"
#include "AdvancedSettingsWindow.hh"
#include "API.hh"

#include "user_interface.hh"

namespace holovibes::gui
{

AdvancedSettingsWindow::AdvancedSettingsWindow(QMainWindow* parent, AdvancedSettingsWindowPanel* specific_panel)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    setWindowIcon(QIcon(":/holovibes_logo.png"));
    this->setAttribute(Qt::WA_DeleteOnClose);
    this->show();

    plug_specific_panel(specific_panel);

    set_current_values();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow() { UserInterface::instance().advanced_settings_window.release(); }

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
    api::set_file_buffer_size(static_cast<int>(ui.FileBSSpinBox->value()));
    api::set_input_buffer_size(ui.InputBSSpinBox->value());
    api::set_record_buffer_size(static_cast<int>(ui.RecordBSSpinBox->value()));
    api::set_output_buffer_size(static_cast<int>(ui.OutputBSSpinBox->value()));
    api::set_time_transformation_cuts_output_buffer_size(static_cast<int>(ui.Cuts3DBSSpinBox->value()));

    api::set_display_rate(ui.DisplayRateSpinBox->value());
    api::change_filter2d_smooth()->low = ui.Filter2DLowSpinBox->value();
    api::change_filter2d_smooth()->high = ui.Filter2DHighSpinBox->value();
    api::change_current_view_as_view_xyz()->contrast.min = ui.ContrastLowerSpinBox->value();
    api::change_current_view_as_view_xyz()->contrast.max = ui.ContrastUpperSpinBox->value();
    api::set_renorm_constant(ui.RenormConstantSpinBox->value());
    api::change_contrast_threshold()->frame_index_offset = ui.CutsContrastSpinBox->value();

    UserInterface::instance().default_output_filename_ = ui.OutputNameLineEdit->text().toStdString();
    UserInterface::instance().record_output_directory_ = ui.InputFolderPathLineEdit->text().toStdString();
    UserInterface::instance().file_input_directory_ = ui.OutputFolderPathLineEdit->text().toStdString();
    UserInterface::instance().batch_input_directory_ = ui.BatchFolderPathLineEdit->text().toStdString();

    UserInterface::instance().auto_scale_point_threshold_ = ui.autoScalePointThresholdSpinBox->value();

    api::set_raw_bitshift(ui.rawBitShiftSpinBox->value());

    if (specific_panel_ != nullptr)
        specific_panel_->set_ui_values();

    this->close();
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
    ui.FileBSSpinBox->setValue(api::get_file_buffer_size());
    ui.InputBSSpinBox->setValue(api::get_input_buffer_size());
    ui.RecordBSSpinBox->setValue(api::get_record_buffer_size());
    ui.OutputBSSpinBox->setValue(api::get_output_buffer_size());
    ui.Cuts3DBSSpinBox->setValue(api::get_time_transformation_cuts_output_buffer_size());

    ui.DisplayRateSpinBox->setValue(api::get_display_rate());
    ui.Filter2DLowSpinBox->setValue(api::get_filter2d_smooth().low);
    ui.Filter2DHighSpinBox->setValue(api::get_filter2d_smooth().high);
    ui.ContrastLowerSpinBox->setValue(api::get_contrast_threshold().lower);
    ui.ContrastUpperSpinBox->setValue(api::get_contrast_threshold().upper);
    ui.RenormConstantSpinBox->setValue(api::get_renorm_constant());
    ui.CutsContrastSpinBox->setValue(api::get_contrast_threshold().frame_index_offset);

    ui.OutputNameLineEdit->setText(UserInterface::instance().default_output_filename_.c_str());
    ui.InputFolderPathLineEdit->setText(UserInterface::instance().record_output_directory_.c_str());
    ui.OutputFolderPathLineEdit->setText(UserInterface::instance().file_input_directory_.c_str());
    ui.BatchFolderPathLineEdit->setText(UserInterface::instance().batch_input_directory_.c_str());

    ui.autoScalePointThresholdSpinBox->setValue(
        static_cast<int>(UserInterface::instance().auto_scale_point_threshold_));

    ui.rawBitShiftSpinBox->setValue(api::get_raw_bitshift());

    if (specific_panel_ != nullptr)
        specific_panel_->set_current_values();
}

} // namespace holovibes::gui
