// Suppress all warnings in this auto-generated file
#pragma warning(push, 0)
#include "ui_analysisplotwindow.h"
#pragma warning(pop)
#include "AnalysisPlotWindow.hh"
#include "API.hh"
#include "concurrent_deque.hh"
#include "GUI.hh"

#define WIDTH 580
#define HEIGHT 250

namespace holovibes::gui
{
AnalysisPlotWindow::AnalysisPlotWindow(ConcurrentDeque<double>& data_vect,
                                       const size_t auto_scale_point_threshold,
                                       const QString title,
                                       QWidget* parent)
    : QMainWindow(parent)
    , curve_plot_(data_vect, auto_scale_point_threshold, title, WIDTH, HEIGHT, this)
{
    ui.setupUi(this);
    resize(WIDTH, HEIGHT);
    this->show();

    unsigned int s = static_cast<unsigned int>(data_vect.size());
    if (s > 0)
        curve_plot_.resize_plot(s);

    QSpinBox* p = findChild<QSpinBox*>("PointsNbSpinBox");
    change_points_nb(p->value());
}

AnalysisPlotWindow::~AnalysisPlotWindow() {}

void AnalysisPlotWindow::closeEvent(QCloseEvent* event) { emit closed(); }

void AnalysisPlotWindow::resizeEvent(QResizeEvent* e)
{
    curve_plot_.resize(e->size().width() - 20, e->size().height() - 50);
}

void AnalysisPlotWindow::start_drawing() { curve_plot_.start(); }

void AnalysisPlotWindow::stop_drawing() { curve_plot_.stop(); }

void AnalysisPlotWindow::auto_scale() { curve_plot_.auto_scale(); }

void AnalysisPlotWindow::change_points_nb(int n) { curve_plot_.set_points_nb(n); }

void AnalysisPlotWindow::change_curve(int curve_to_plot) { curve_plot_.change_curve(curve_to_plot); }
} // namespace holovibes::gui

#undef WIDTH
#undef HEIGHT
