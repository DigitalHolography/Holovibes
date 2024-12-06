/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

namespace holovibes
{
/*! \class ChartPoint
 *
 * \brief #TODO Add a description for this struct
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
