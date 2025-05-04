/*! \file
 *
 * \brief Declaration of the ChartPoint struct
 */
#pragma once

namespace holovibes
{
/*! \class ChartPoint
 *
 * \brief Struct that holds the data for a single point on a chart
 */
struct ChartPoint
{
    double avg_signal;
    double avg_noise;
    double avg_signal_div_avg_noise;
    double log_avg_signal_div_avg_noise;
    double std_signal;
    double std_signal_div_avg_noise;
    double std_signal_div_avg_signal;
};

/*! \class ChartMeanVesselsPoint
 *
 * \brief struct use for showing Chart Mean Vessels
 */
struct ChartMeanVesselsPoint
{
    double mean_artery;
    double mean_veins;
    double mean_choroid;

    double min() const { return std::min(mean_artery, std::min(mean_veins, mean_choroid)); }

    double max() const { return std::max(mean_artery, std::max(mean_veins, mean_choroid)); }

    bool operator<(const ChartMeanVesselsPoint& rhs) const { return this->min() < rhs.min(); }

    bool operator>(const ChartMeanVesselsPoint& rhs) const { return this->max() > rhs.max(); }
};

} // namespace holovibes
