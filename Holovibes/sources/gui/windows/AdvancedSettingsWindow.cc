#include "ui_advancedsettingswindow.h"
#include "AdvancedSettingsWindow.hh"

namespace holovibes::gui
{
AdvancedSettingsWindow::AdvancedSettingsWindow(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    this->show();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow() {}

void AdvancedSettingsWindow::closeEvent(QCloseEvent* event) { emit closed(); }
} // namespace holovibes::gui
