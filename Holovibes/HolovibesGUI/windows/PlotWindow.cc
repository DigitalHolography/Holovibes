// Suppress all warnings in this auto-generated file
#pragma warning(push, 0)
#include "ui_plotwindow.h"
#pragma warning(pop)
#include "PlotWindow.hh"
#include "concurrent_deque.hh"

#define WIDTH 580
#define HEIGHT 250

namespace holovibes::gui
{
PlotWindow::PlotWindow(ConcurrentDeque<ChartPoint>& data_vect,
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

PlotWindow::~PlotWindow() {}

void PlotWindow::closeEvent(QCloseEvent* event) { emit closed(); }

void PlotWindow::resizeEvent(QResizeEvent* e) { curve_plot_.resize(e->size().width() - 20, e->size().height() - 50); }

void PlotWindow::start_drawing() { curve_plot_.start(); }

void PlotWindow::stop_drawing() { curve_plot_.stop(); }

void PlotWindow::auto_scale() { curve_plot_.auto_scale(); }

void PlotWindow::change_points_nb(int n) { curve_plot_.set_points_nb(n); }

void PlotWindow::change_curve(int curve_to_plot) { curve_plot_.change_curve(curve_to_plot); }

void PlotWindow::toggle_dark_mode()
{
    QCheckBox* box = findChild<QCheckBox*>("darkModeCheckBox");
    curve_plot_.toggle_dark_mode(box->isChecked());
}
} // namespace holovibes::gui

#undef WIDTH
#undef HEIGHT
