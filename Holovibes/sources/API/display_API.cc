#include "API.hh"

namespace holovibes::api
{

const ViewWindow& get_view(WindowKind kind)
{
    if (kind == WindowKind::ViewXY)
        return static_cast<const ViewWindow&>(GSH::instance().get_view_cache().get_value<ViewXY>());
    else if (kind == WindowKind::ViewXZ)
        return static_cast<const ViewWindow&>(GSH::instance().get_view_cache().get_value<ViewXZ>());
    else if (kind == WindowKind::ViewYZ)
        return static_cast<const ViewWindow&>(GSH::instance().get_view_cache().get_value<ViewYZ>());
    else if (kind == WindowKind::ViewFilter2D)
        return static_cast<const ViewWindow&>(GSH::instance().get_view_cache().get_value<ViewFilter2D>());

    throw std::runtime_error("Unexpected WindowKind");
    // default case
    return static_cast<const ViewWindow&>(GSH::instance().get_view_cache().get_value<ViewXY>());
}

TriggerChangeValue<ViewWindow> change_view(WindowKind kind)
{
    if (kind == WindowKind::ViewXY)
        return static_cast<TriggerChangeValue<ViewWindow>>(GSH::instance().get_view_cache().change_value<ViewXY>());
    else if (kind == WindowKind::ViewXZ)
        return static_cast<TriggerChangeValue<ViewWindow>>(GSH::instance().get_view_cache().change_value<ViewXZ>());
    else if (kind == WindowKind::ViewYZ)
        return static_cast<TriggerChangeValue<ViewWindow>>(GSH::instance().get_view_cache().change_value<ViewYZ>());
    else if (kind == WindowKind::ViewFilter2D)
        return static_cast<TriggerChangeValue<ViewWindow>>(
            GSH::instance().get_view_cache().change_value<ViewFilter2D>());

    throw std::runtime_error("Unexpected WindowKind");
    return TriggerChangeValue<ViewWindow>([]() {}, nullptr);
}

} // namespace holovibes::api