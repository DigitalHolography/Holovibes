#include "view_gsh_on_change.hh"
#include "API.hh"
#include "user_interface.hh"

namespace holovibes
{
static void check_limits_pq(ViewAccuPQ& view_pq)
{
    int upper_bound = api::get_time_transformation_size() - 1;

    if (upper_bound == 0)
    {
        view_pq.start = 0;
        view_pq.width = 0;
        return;
    }

    if (view_pq.width > upper_bound)
        view_pq.width = upper_bound;

    upper_bound -= view_pq.width;

    if (upper_bound > 0 && view_pq.start > static_cast<uint>(upper_bound))
        view_pq.start = upper_bound;
    else if (upper_bound == 0)
    {
        view_pq.start = 0;
    }
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
bool ViewGSHOnChange::change_accepted<ViewAccuX>(const ViewAccuXY& new_value)
{
    return new_value.start <= api::get_import_frame_descriptor().width;
}

template <>
bool ViewGSHOnChange::change_accepted<ViewAccuY>(const ViewAccuXY& new_value)
{
    return new_value.start <= api::get_import_frame_descriptor().height;
}

template <>
void ViewGSHOnChange::operator()<ViewAccuX>(ViewAccuXY& new_value)
{
    LOG_UPDATE_ON_CHANGE(ViewAccuX);
}

template <>
void ViewGSHOnChange::operator()<ViewAccuY>(ViewAccuXY& new_value)
{
    LOG_UPDATE_ON_CHANGE(ViewAccuY);
}

template <>
void ViewGSHOnChange::operator()<CutsViewEnabled>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(CutsViewEnabled);

    if (new_value)
    {
        api::detail::set_value<TimeTransformationCutsEnable>(true);
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXZ);
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewYZ);
    }
}

template <>
void ViewGSHOnChange::operator()<Filter2DViewEnabled>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(Filter2DViewEnabled);
    if (new_value)
    {
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewFilter2D);
    }
}

template <>
void ViewGSHOnChange::operator()<LensViewEnabled>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(LensViewEnabled);
}

template <>
void ViewGSHOnChange::operator()<RawViewEnabled>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(RawViewEnabled);
}

template <>
void ViewGSHOnChange::operator()<ChartDisplayEnabled>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(ChartDisplayEnabled);
}

template <>
bool ViewGSHOnChange::change_accepted<RawViewEnabled>(bool new_value)
{
    return !(new_value && api::get_batch_size() > api::get_gpu_output_queue().get_size());
}

template <>
bool ViewGSHOnChange::change_accepted<ViewAccuP>(const ViewAccuPQ& new_value)
{
    if (new_value.start + new_value.width >= api::get_time_transformation_size())
        return false;
    return true;
}

template <>
bool ViewGSHOnChange::change_accepted<Filter2DViewEnabled>(bool new_value)
{
    return true;
}

template <>
bool ViewGSHOnChange::change_accepted<CutsViewEnabled>(bool new_value)
{
    return true;
}

template <>
bool ViewGSHOnChange::change_accepted<ChartDisplayEnabled>(bool new_value)
{
    return true;
}

template <>
bool ViewGSHOnChange::change_accepted<LensViewEnabled>(bool new_value)
{
    return true;
}

} // namespace holovibes