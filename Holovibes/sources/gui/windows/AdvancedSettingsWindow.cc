#include "AdvancedSettingsWindow.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include "QPathSelectorLayout.hh"
#include "API.hh"
#include "asw_panel_buffer_size.hh"

namespace holovibes::gui
{

QGroupBox* AdvancedSettingsWindow::create_group_box(const std::string& name)
{
    QGroupBox* group_box = new QGroupBox(this);
    group_box->setTitle(QString::fromUtf8(name.c_str()));
    return group_box;
}

QGroupBox* AdvancedSettingsWindow::create_advanced_group_box(const std::string& name)
{
    QGroupBox* advanced_group_box = create_group_box(name);

    QVBoxLayout* advanced_layout = new QVBoxLayout(this);

    // File spin box
    QDoubleSpinBoxLayout* display_rate = new QDoubleSpinBoxLayout(this, main_widget, "DisplayRate");
    display_rate->setValue(0);
    advanced_layout->addItem(display_rate);

    // Input spin box
    QDoubleSpinBoxLayout* filter2d_smooth_low = new QDoubleSpinBoxLayout(this, main_widget, "Filter2D_smooth_low");
    filter2d_smooth_low->setValue(0);
    advanced_layout->addItem(filter2d_smooth_low);

    // Input spin box
    QDoubleSpinBoxLayout* filter2d_smooth_high = new QDoubleSpinBoxLayout(this, main_widget, "Filter2D_smooth_high");
    filter2d_smooth_high->setValue(0.5f);
    advanced_layout->addItem(filter2d_smooth_high);

    // Record spin box
    QDoubleSpinBoxLayout* contrast_upper_threshold =
        new QDoubleSpinBoxLayout(this, main_widget, "Contrast_upper_threshold");
    contrast_upper_threshold->setValue(99.5f);
    advanced_layout->addItem(contrast_upper_threshold);

    // Output spin box
    QIntSpinBoxLayout* renorm_constant = new QIntSpinBoxLayout(this, main_widget, "Renorm_constant");
    renorm_constant->setValue(5);
    advanced_layout->addItem(renorm_constant);

    // 3D cuts spin box
    QIntSpinBoxLayout* cuts_contrast_p_offset = new QIntSpinBoxLayout(this, main_widget, "Cuts_contrast_p_offset");
    cuts_contrast_p_offset->setValue(0);
    advanced_layout->addItem(cuts_contrast_p_offset);

    advanced_group_box->setLayout(advanced_layout);

    return advanced_group_box;
}

QGroupBox* AdvancedSettingsWindow::create_file_group_box(const std::string& name)
{
    QGroupBox* file_group_box = create_group_box(name);

    QVBoxLayout* file_layout = new QVBoxLayout(this);

    // Default input folder path selector
    QPathSelectorLayout* default_input_folder = new QPathSelectorLayout(this, main_widget);
    default_input_folder->setName("Default Input folder")->setText("file1");
    file_layout->addItem(default_input_folder);

    // Default output folder path selector
    QPathSelectorLayout* default_output_folder = new QPathSelectorLayout(this, main_widget);
    default_output_folder->setName("Default Output folder")->setText("file2");
    file_layout->addItem(default_output_folder);

    // Batch input folder path selector
    QPathSelectorLayout* batch_input_folder = new QPathSelectorLayout(this, main_widget);
    batch_input_folder->setName("Batch Input folder")->setText("file3");
    file_layout->addItem(batch_input_folder);

    file_group_box->setLayout(file_layout);

    return file_group_box;
}

QGroupBox* AdvancedSettingsWindow::create_chart_group_box(const std::string& name)
{
    QGroupBox* chart_group_box = create_group_box(name);

    QVBoxLayout* chart_layout = new QVBoxLayout(this);

    // File spin box
    QIntSpinBoxLayout* auto_scale_point_threshold = new QIntSpinBoxLayout(this, main_widget, "DisplayRate");
    auto_scale_point_threshold->setValue(100);
    chart_layout->addItem(auto_scale_point_threshold);

    chart_group_box->setLayout(chart_layout);

    return chart_group_box;
}

AdvancedSettingsWindow::AdvancedSettingsWindow(QMainWindow* parent)
    : QMainWindow(parent)
{
    this->setWindowTitle("AdvancedSettings");
    main_widget = new QWidget(this);
    main_layout = new QHBoxLayout(this);
    main_widget->setLayout(main_layout);

    // ################################################################################################
    ASWPanelBufferSize* buffer_size_panel = new ASWPanelBufferSize(this, main_widget);
    main_layout->addWidget(buffer_size_panel);

    QGroupBox* advanced_group_box = create_advanced_group_box("Advanced");
    main_layout->addWidget(advanced_group_box);

    QGroupBox* file_group_box = create_file_group_box("File");
    main_layout->addWidget(file_group_box);

    QGroupBox* chart_group_box = create_chart_group_box("Chart");
    main_layout->addWidget(chart_group_box);
    // ################################################################################################

    setCentralWidget(main_widget);
    this->show();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow() {}

void AdvancedSettingsWindow::closeEvent(QCloseEvent* event) { emit closed(); }
} // namespace holovibes::gui
