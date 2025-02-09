#include "gui_analysis_curve_plot.hh"
#include "chart_point.hh"

#include <QLegendMarker>

#define WIDTH 600
#define HEIGHT 300
#define TOP_OFFSET 30
#define SIZE_OFFSET 20
#define TIMER_FREQ 40
#define POINTS 200

#define COLOR_ARTERY QColor::fromRgb(235, 64, 52)
#define COLOR_VEIN QColor::fromRgb(63, 42, 247)
#define COLOR_CHOROID QColor::fromRgb(0, 192, 0)

#include <iostream>

namespace holovibes::gui
{
AnalysisCurvePlot::AnalysisCurvePlot(ConcurrentDeque<ChartMeanVesselsPoint>& data_vect,
                                     const size_t auto_scale_point_threshold,
                                     const QString title,
                                     const unsigned int width,
                                     const unsigned int height,
                                     QWidget* parent)
    : QWidget(parent)
    , data_vect_(data_vect)
    , points_nb_(POINTS)
    , timer_(this)
    , current_curve_(AnalysisCurveName::VEIN_MEAN)
    , auto_scale_point_threshold_(auto_scale_point_threshold)
    , auto_scale_curr_points_(0)
{
    line_series_1 = new QLineSeries();
    line_series_2 = new QLineSeries();
    line_series_3 = new QLineSeries();
    chart = new QChart();
    chart_view = new QChartView(chart, parent);

    chart->setTheme(QChart::ChartTheme::ChartThemeDark);
    line_series_1->setColor(COLOR_ARTERY);
    line_series_2->setColor(COLOR_VEIN);
    line_series_3->setColor(COLOR_CHOROID);

    chart->legend()->hide();
    chart->addSeries(line_series_1);
    chart->addSeries(line_series_2);
    chart->addSeries(line_series_3);
    chart->createDefaultAxes();
    chart->legend()->setVisible(false);
    chart->legend()->setAlignment(Qt::AlignBottom);

    QList<QLegendMarker*> list = chart->legend()->markers();
    // std::cout << "Labels: " << list[0]->label().toStdString() << std::endl;
    list[0]->setLabel("Arteries");
    list[1]->setLabel("Veins");
    list[2]->setLabel("Choroid");

    // reverse the x axis
    // chart->axisX()->setRange(0, points_nb_);
    // chart->axisX()->setReverse(true);

    // FIXME axisX is deprecated, axes returns a list of axes, we take the first horizontal, verify ?
    chart->axes(Qt::Horizontal).front()->setRange(0, points_nb_);
    chart->axes(Qt::Horizontal).front()->setReverse(true);

    chart->setTitle(title);

    chart_view->setRenderHint(QPainter::Antialiasing);
    chart_view->move(0, TOP_OFFSET);

    change_curve(AnalysisCurveName::ARTERY_MEAN);

    chart_vector_.resize(points_nb_);
    connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
    timer_.start(TIMER_FREQ);
}

AnalysisCurvePlot::~AnalysisCurvePlot()
{
    delete line_series_1;
    delete line_series_2;
    delete line_series_3;
    delete chart;
    delete chart_view;
}

void AnalysisCurvePlot::change_curve(AnalysisCurveName curve_to_plot)
{
    current_curve_ = curve_to_plot;
    chart->legend()->setVisible(false);
    request_rescale();
    switch (current_curve_)
    {
    case AnalysisCurveName::ARTERY_MEAN:
        curve_get_ = [](const ChartMeanVesselsPoint& point) { return point.mean_artery; };
        break;
    case AnalysisCurveName::VEIN_MEAN:
        curve_get_ = [](const ChartMeanVesselsPoint& point) { return point.mean_veins; };
        break;
    case AnalysisCurveName::CHOROID_MEAN:
        curve_get_ = [](const ChartMeanVesselsPoint& point) { return point.mean_choroid; };
        break;
    case AnalysisCurveName::ALL_MEAN:
        chart->legend()->setVisible(true);
        break;
    default:
        abort();
    }
}

QSize AnalysisCurvePlot::minimumSizeHint() const { return QSize(WIDTH, HEIGHT); }

QSize AnalysisCurvePlot::sizeHint() const { return QSize(WIDTH, HEIGHT); }

void AnalysisCurvePlot::resize_plot(const int size)
{
    points_nb_ = size;
    chart_vector_.resize(size);
    // chart->axisX()->setMax(QVariant(points_nb_)); // warning C4996: 'QChart::axisX': was declared deprecated

    chart->axes(Qt::Horizontal).front()->setMax(QVariant(points_nb_));
}

void AnalysisCurvePlot::resizeEvent(QResizeEvent* e)
{
    chart_view->resize(e->size().width() + SIZE_OFFSET, e->size().height() + SIZE_OFFSET);
}

void AnalysisCurvePlot::load_data_vector()
{
    AnalysisCurveName curve_type = current_curve_; // maybe stops thread undefined behaviours ?

    QList<QPointF> new_data_1;
    QList<QPointF> new_data_2;
    QList<QPointF> new_data_3;

    if (!data_vect_.empty())
    {
        size_t copied_elts_nb = data_vect_.fill_array(chart_vector_, points_nb_);
        new_data_1.reserve(static_cast<int>(copied_elts_nb));
        if (curve_type == AnalysisCurveName::ALL_MEAN)
        {
            new_data_2.reserve(static_cast<int>(copied_elts_nb));
            new_data_3.reserve(static_cast<int>(copied_elts_nb));
        }

        ++auto_scale_curr_points_;

        for (size_t i = 0; i < copied_elts_nb; ++i)
        {
            float x = i;
            if (curve_type != AnalysisCurveName::ALL_MEAN)
                new_data_1.push_back(QPointF(x, curve_get_(chart_vector_[i])));
            else
            {
                new_data_1.push_back(QPointF(x, chart_vector_[i].mean_artery));
                new_data_2.push_back(QPointF(x, chart_vector_[i].mean_veins));
                new_data_3.push_back(QPointF(x, chart_vector_[i].mean_choroid));
            }
        }

        if (auto_scale_curr_points_ > auto_scale_point_threshold_)
        {
            auto_scale_curr_points_ -= auto_scale_point_threshold_;
            auto_scale();
        }
    }

    line_series_1->replace(new_data_1);
    line_series_2->replace(new_data_2);
    line_series_3->replace(new_data_3);
}

void AnalysisCurvePlot::request_rescale() { auto_scale_curr_points_ = auto_scale_point_threshold_; }

void AnalysisCurvePlot::auto_scale()
{
    std::vector<ChartMeanVesselsPoint> tmp = chart_vector_;

    double min;
    double max;
    if (current_curve_ == AnalysisCurveName::ALL_MEAN)
    {
        min = std::min_element(tmp.cbegin(),
                               tmp.cend(),
                               [&](const ChartMeanVesselsPoint& lhs, const ChartMeanVesselsPoint& rhs)
                               { return lhs < rhs; })
                  ->min();
        max = std::min_element(tmp.cbegin(),
                               tmp.cend(),
                               [&](const ChartMeanVesselsPoint& lhs, const ChartMeanVesselsPoint& rhs)
                               { return lhs > rhs; })
                  ->max();
    }
    else
    {
        auto minmax = std::minmax_element(tmp.cbegin(),
                                          tmp.cend(),
                                          [&](const ChartMeanVesselsPoint& lhs, const ChartMeanVesselsPoint& rhs)
                                          { return curve_get_(lhs) < curve_get_(rhs); });
        min = curve_get_(*(minmax.first));
        max = curve_get_(*(minmax.second));
    }

    double offset = (max - min) / 10.0;

    chart->axes(Qt::Vertical).front()->setRange(min - offset, max + offset);
}

void AnalysisCurvePlot::start() { timer_.start(TIMER_FREQ); }

void AnalysisCurvePlot::stop()
{
    if (timer_.isActive())
        timer_.stop();
}

void AnalysisCurvePlot::set_points_nb(const unsigned int n)
{
    stop();
    resize_plot(n);
    start();
}

void AnalysisCurvePlot::update()
{
    load_data_vector();
    while (data_vect_.size() > points_nb_)
        data_vect_.pop_front();
}

} // namespace holovibes::gui
#undef WIDTH
#undef HEIGHT
#undef TOP_OFFSET
#undef SIZE_OFFSET
#undef TIMER_FREQ
#undef POINTS
