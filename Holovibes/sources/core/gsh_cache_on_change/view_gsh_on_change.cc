#include "view_gsh_on_change.hh"
#include "API.hh"

namespace holovibes
{
static void check_limits_pq(ViewAccuPQ& view_pq)
{
    int upper_bound = api::get_time_transformation_size() - 1;

    if (view_pq.width > upper_bound)
        view_pq.width = upper_bound;

    upper_bound -= view_pq.width;

    if (upper_bound > 0 && view_pq.start > static_cast<uint>(upper_bound))
        view_pq.start = upper_bound;
}

template <>
void ViewGSHOnChange::operator()<ViewAccuP>(ViewAccuPQ& new_value)
{
    LOG_UPDATE_ON_CHANGE(ViewAccuP);

    check_limits_pq(new_value);
}
template <>
void ViewGSHOnChange::operator()<ViewAccuQ>(ViewAccuPQ& new_value)
{
    LOG_UPDATE_ON_CHANGE(ViewAccuQ);

    check_limits_pq(new_value);
}

template <>
void ViewGSHOnChange::operator()<ViewAccuX>(ViewAccuXY& new_value)
{
    LOG_UPDATE_ON_CHANGE(ViewAccuX);

    if (new_value.start > api::get_import_frame_descriptor().width)
    {
        LOG_WARN("New X start value was rejected because it's out of bound");
        new_value.start = api::detail::get_value<ViewAccuX>().start;
    }
}

template <>
void ViewGSHOnChange::operator()<ViewAccuY>(ViewAccuXY& new_value)
{
    LOG_UPDATE_ON_CHANGE(ViewAccuY);

    if (new_value.start > api::get_import_frame_descriptor().height)
    {
        LOG_WARN("New X start value was rejected because it's out of bound");
        new_value.start = api::detail::get_value<ViewAccuY>().start;
    }
}

template <>
void ViewGSHOnChange::operator()<CutsViewEnable>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(CutsViewEnable);

    api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXZ);
    api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewYZ);

    if (new_value)
        api::detail::set_value<TimeTransformationCutsEnable>(true);
}

template <>
void ViewGSHOnChange::operator()<LensViewEnabled>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(LensViewEnabled);

    if (api::get_compute_mode() == ComputeModeEnum::Raw)
        new_value = false;
}

template <>
void ViewGSHOnChange::operator()<RawViewEnabled>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(RawViewEnabled);

    if (api::get_compute_mode() == ComputeModeEnum::Raw)
        new_value = false;
}

template <>
bool ViewGSHOnChange::change_accepted<RawViewEnabled>(bool new_value)
{
    return !(new_value && api::get_batch_size() > api::get_gpu_output_queue().get_size());
}
} // namespace holovibes