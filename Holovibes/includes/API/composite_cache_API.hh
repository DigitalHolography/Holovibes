#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

// notify()

inline CompositeKind get_composite_kind() { return api::detail::get_value<CompositeKindParam>(); }
inline void set_composite_kind(CompositeKind value) { api::detail::set_value<CompositeKindParam>(value); }

inline const CompositeRGB get_composite_rgb() { return api::detail::get_value<CompositeRGBParam>(); }
inline const CompositeHSV get_CompositeHsv() { return api::detail::get_value<CompositeHSVParam>(); }

inline CompositeRGB& change_composite_rgb() { return api::detail::change_value<CompositeRGBParam>(); }
inline CompositeHSV& change_CompositeHsv() { return api::detail::change_value<CompositeHSVParam>(); }

inline const CompositeP get_composite_p()
{
    if (api::get_composite_kind() == CompositeKind::RGB)
        return get_composite_rgb();
    return get_CompositeHsv();
}

inline CompositeP& change_composite_p()
{
    if (api::get_composite_kind() == CompositeKind::RGB)
        return change_composite_rgb();
    return change_CompositeHsv();
}

} // namespace holovibes::api
