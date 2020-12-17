/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

namespace holovibes
{
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