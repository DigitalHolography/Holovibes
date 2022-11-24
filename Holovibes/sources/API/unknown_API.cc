#include "API.hh"

namespace holovibes::api
{

void check_p_limits()
{
    int upper_bound = get_time_transformation_size() - 1;

    if (api::get_view_accu_p().accu_level > upper_bound)
        api::change_view_accu_p()->accu_level = upper_bound;

    upper_bound -= api::get_view_accu_p().accu_level;

    if (upper_bound >= 0 && api::get_view_accu_p().index > static_cast<uint>(upper_bound))
        api::change_view_accu_p()->index = upper_bound;
}

void check_q_limits()
{
    int upper_bound = get_time_transformation_size() - 1;

    if (api::get_view_accu_q().accu_level > upper_bound)
        api::change_view_accu_q()->accu_level = upper_bound;

    upper_bound -= api::get_view_accu_q().accu_level;

    if (upper_bound >= 0 && api::get_view_accu_q().index > static_cast<uint>(upper_bound))
        api::change_view_accu_q()->index = upper_bound;
}

void close_critical_compute()
{
    api::change_convolution()->enabled = false;

    if (api::get_cuts_view_enabled())
        cancel_time_transformation_cuts([]() {});

    Holovibes::instance().stop_compute();
}

bool slide_update_threshold(
    const int slider_value, float& receiver, float& bound_to_update, const float lower_bound, const float upper_bound)
{
    receiver = slider_value / 1000.0f;

    if (lower_bound > upper_bound)
    {
        // FIXME bound_to_update = receiver ?
        bound_to_update = slider_value / 1000.0f;

        return true;
    }

    return false;
}

void set_log_scale(const bool value) { api::change_current_view()->log_enabled = value; }

} // namespace holovibes::api
