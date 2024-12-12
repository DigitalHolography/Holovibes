/*! \file
 *
 * \brief definition of struct use for chart
 */
#pragma once

namespace holovibes
{
/*! \class ChartPoint
 *
 * \brief struct use for export chart
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
};

} // namespace holovibes
