#include "AdvancedSettingsWindow.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include "QPathSelectorLayout.hh"
#include "API.hh"
#include "asw_panel_buffer_size.hh"
#include "asw_panel_advanced.hh"
#include "asw_panel_file.hh"

namespace holovibes::gui
{

QGroupBox* AdvancedSettingsWindow::create_group_box(const std::string& name)
{
    QGroupBox* group_box = new QGroupBox(this);
    group_box->setTitle(QString::fromUtf8(name.c_str()));
    return group_box;
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

    ASWPanelFile* file_panel = new ASWPanelFile(this, main_widget);
    main_layout->addWidget(file_panel);

    QGroupBox* chart_group_box = create_chart_group_box("Chart");
    main_layout->addWidget(chart_group_box);
    // ################################################################################################

    setCentralWidget(main_widget);
    this->show();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow() {}

void AdvancedSettingsWindow::closeEvent(QCloseEvent* event) { emit closed(); }
} // namespace holovibes::gui
