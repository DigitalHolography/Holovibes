/*! \file
 *
 * \brief declaration of the ChartPoint struct
 */
#pragma once

namespace holovibes
{
/*! \class ChartPoint
 *
 * \brief struct that holds the data for a single point on a chart
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
} // namespace holovibes
