#include "ui_advancedsettingswindow.h"
#include "AdvancedSettingsWindow.hh"
#include "MainWindow.hh"

namespace holovibes::gui
{

AdvancedSettingsWindow::AdvancedSettingsWindow(ComputeDescriptor& cd, MainWindow* parent)
    : QMainWindow((QMainWindow)parent)
    , cd_(cd)
    , mainwindow_(*parent)
{
    ui.setupUi(this);
    setWindowIcon(QIcon("Holovibes.ico"));
    this->show();

    set_current_values();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow() {}

void AdvancedSettingsWindow::closeEvent(QCloseEvent* event) { emit closed(); }

void AdvancedSettingsWindow::set_current_values()
{
    ui.FileBSSpinBox->setValue(cd_.file_buffer_size);
    ui.InputBSSpinBox->setValue(cd_.input_buffer_size);
    ui.RecordBSSpinBox->setValue(cd_.record_buffer_size);
    ui.OutputBSSpinBox->setValue(cd_.output_buffer_size);
    ui.Cuts3DBSSpinBox->setValue(cd_.time_transformation_cuts_output_buffer_size);

    ui.DisplayRateSpinBox->setValue(cd_.display_rate);
    ui.Filter2DLowSpinBox->setValue(cd_.filter2d_smooth_low);
    ui.Filter2DHighSpinBox->setValue(cd_.filter2d_smooth_high);
    ui.ContrastLowerSpinBox->setValue(cd_.contrast_lower_threshold);
    ui.ContrastUpperSpinBox->setValue(cd_.contrast_upper_threshold);
    ui.RenormConstantSpinBox->setValue(cd_.renorm_constant);
    ui.CutsContrastSpinBox->setValue(cd_.cuts_contrast_p_offset);

    ui.OutputNameLineEdit->setText(mainwindow_.default_output_filename_.c_str());
    ui.InputFolderPathLineEdit->setText(mainwindow_.record_output_directory_.c_str());
    ui.OutputFolderPathLineEdit->setText(mainwindow_.file_input_directory_.c_str());
    ui.BatchFolderPathLineEdit->setText(mainwindow_.batch_input_directory_.c_str());

    ui.autoScalePointThresholdSpinBox->setValue(mainwindow_.auto_scale_point_threshold_);

    ui.ZStepSpinBox->setValue(mainwindow_.z_step_);
    ui.RecordFrameStepSpinBox->setValue(mainwindow_.record_frame_step_);

    ui.ReloadLabel->setVisible(false);
}

void AdvancedSettingsWindow::set_ui_values()
{
    cd_.file_buffer_size = ui.FileBSSpinBox->value();
    cd_.input_buffer_size = ui.InputBSSpinBox->value();
    cd_.record_buffer_size = ui.RecordBSSpinBox->value();
    cd_.output_buffer_size = ui.OutputBSSpinBox->value();
    cd_.time_transformation_cuts_output_buffer_size = ui.Cuts3DBSSpinBox->value();

    cd_.display_rate = ui.DisplayRateSpinBox->value();
    cd_.filter2d_smooth_low = ui.Filter2DLowSpinBox->value();
    cd_.filter2d_smooth_high = ui.Filter2DHighSpinBox->value();
    cd_.contrast_lower_threshold = ui.ContrastLowerSpinBox->value();
    cd_.contrast_upper_threshold = ui.ContrastUpperSpinBox->value();
    cd_.renorm_constant = ui.RenormConstantSpinBox->value();
    cd_.cuts_contrast_p_offset = ui.CutsContrastSpinBox->value();

    mainwindow_.default_output_filename_ = ui.OutputNameLineEdit->text().toStdString();
    mainwindow_.record_output_directory_ = ui.InputFolderPathLineEdit->text().toStdString();
    mainwindow_.file_input_directory_ = ui.OutputFolderPathLineEdit->text().toStdString();
    mainwindow_.batch_input_directory_ = ui.BatchFolderPathLineEdit->text().toStdString();

    mainwindow_.auto_scale_point_threshold_ = ui.autoScalePointThresholdSpinBox->value();

    mainwindow_.z_step_ = ui.ZStepSpinBox->value();
    mainwindow_.record_frame_step_ = ui.RecordFrameStepSpinBox->value();

    ui.ReloadLabel->setVisible(true);
    mainwindow_.need_close = true;
}

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

void AdvancedSettingsWindow::change_input_folder_path() { change_folder(ui.InputFolderPathLineEdit); }
void AdvancedSettingsWindow::change_output_folder_path() { change_folder(ui.OutputFolderPathLineEdit); }
void AdvancedSettingsWindow::change_batch_input_folder_path() { change_folder(ui.BatchFolderPathLineEdit); }
} // namespace holovibes::gui
