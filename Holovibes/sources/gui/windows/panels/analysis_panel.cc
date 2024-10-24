/*! \file
 *
 */

#include <filesystem>

#include "analysis_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "API.hh"

namespace api = ::holovibes::api;

namespace holovibes::gui
{
AnalysisPanel::AnalysisPanel(QWidget* parent)
    : Panel(parent)
{
}

AnalysisPanel::~AnalysisPanel() {}

void AnalysisPanel::init() {}

void AnalysisPanel::on_notify() {}

} // namespace holovibes::gui