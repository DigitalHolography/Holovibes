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

AdvancedSettingsWindow::AdvancedSettingsWindow(QMainWindow* parent, AdvancedSettingsWindowPanel* specific_panel)
    : QMainWindow(parent)
{
    LOG_ERROR;

    this->setWindowTitle("AdvancedSettings");
    this->setAttribute(Qt::WA_DeleteOnClose);

    // We cannot give a customized layout to a QMainWindow so we have
    // to instantiate an invisible widget that will carry the customized
    // layout
    main_widget_ = new QWidget(this);

    // The customized layout
    main_layout_ = new QGridLayout(main_widget_);

    // Creation of customized pannels
    create_buffer_size_panel();
    create_advanced_panel();
    create_file_panel();
    create_chart_panel();
    plug_specific_panel(specific_panel);

    // Give to the QMainWindow the invisible widget carrying the customized layout
    setCentralWidget(main_widget_);
    this->show();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow()
{
    LOG_INFO;
    UserInterfaceDescriptor::instance().advanced_settings_window_.release();
}

#pragma region PANELS

void AdvancedSettingsWindow::create_buffer_size_panel()
{
    ASWPanelBufferSize* buffer_size_panel = new ASWPanelBufferSize();
    // addWidget(*Widget, row, column, rowspan, colspan)
    main_layout_->addWidget(buffer_size_panel, 1, 0, 1, 1);
}

void AdvancedSettingsWindow::create_advanced_panel()
{
    ASWPanelAdvanced* advanced_panel = new ASWPanelAdvanced();
    // addWidget(*Widget, row, column, rowspan, colspan)
    main_layout_->addWidget(advanced_panel, 1, 1, 1, 1);
}

void AdvancedSettingsWindow::create_file_panel()
{
    ASWPanelFile* file_panel = new ASWPanelFile();
    // addWidget(*Widget, row, column, rowspan, colspan)
    main_layout_->addWidget(file_panel, 0, 0, 1, 2);
}

void AdvancedSettingsWindow::create_chart_panel()
{
    ASWPanelChart* chart_panel = new ASWPanelChart();
    // addWidget(*Widget, row, column, rowspan, colspan)
    main_layout_->addWidget(chart_panel, 2, 0, 1, 1);
}

void AdvancedSettingsWindow::plug_specific_panel(AdvancedSettingsWindowPanel* specific_panel)
{
    if (specific_panel == nullptr)
        return;

    main_layout_->addWidget(specific_panel, 2, 1, 1, 1);
}

#pragma endregion

#pragma region SLOTS

void AdvancedSettingsWindow::closeEvent(QCloseEvent* event) { emit closed(); }

#pragma endregion

} // namespace holovibes::gui
