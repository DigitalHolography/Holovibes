#include "AdvancedSettingsWindow.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include "QPathSelectorLayout.hh"
#include "API.hh"
#include "asw_panel_buffer_size.hh"
#include "asw_panel_advanced.hh"

namespace holovibes::gui
{

QGroupBox* AdvancedSettingsWindow::create_group_box(const std::string& name)
{
    QGroupBox* group_box = new QGroupBox(this);
    group_box->setTitle(QString::fromUtf8(name.c_str()));
    return group_box;
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

    ASWPanelAdvanced* advanced_panel = new ASWPanelAdvanced(this, main_widget);
    main_layout->addWidget(advanced_panel);

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
