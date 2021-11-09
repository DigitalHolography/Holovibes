#include "AdvancedSettingsWindow.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include "QPathSelectorLayout.hh"
#include "API.hh"
#include "asw_panel_buffer_size.hh"
#include "asw_panel_advanced.hh"
#include "asw_panel_file.hh"
#include "asw_panel_chart.hh"

namespace holovibes::gui
{

AdvancedSettingsWindow::AdvancedSettingsWindow(QMainWindow* parent)
    : QMainWindow(parent)
{
    this->setWindowTitle("AdvancedSettings");
    // We cannot give a customized layout to a QMainWindow so we have
    // to instantiate an invisible widget that will carry the customized
    // layout
    main_widget = new QWidget(this);

    // The customized layout
    main_layout = new QHBoxLayout(main_widget);

    // Give to the invisible layout our customized layout
    // main_widget->setLayout(main_layout);

    // Creation of customized pannels
    // ################################################################################################
    ASWPanelBufferSize* buffer_size_panel = new ASWPanelBufferSize(this, main_widget);
    main_layout->addWidget(buffer_size_panel);

    ASWPanelAdvanced* advanced_panel = new ASWPanelAdvanced(this, main_widget);
    main_layout->addWidget(advanced_panel);

    ASWPanelFile* file_panel = new ASWPanelFile(this, main_widget);
    main_layout->addWidget(file_panel);

    ASWPanelChart* chart_panel = new ASWPanelChart(this, main_widget);
    main_layout->addWidget(chart_panel);

    // ################################################################################################

    // Give to the QMainWindow the invisible widget carrying the customized layout
    setCentralWidget(main_widget);
    this->show();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow() {}

void AdvancedSettingsWindow::closeEvent(QCloseEvent* event) { emit closed(); }
} // namespace holovibes::gui
