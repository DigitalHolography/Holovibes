#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

// notify()

inline CompositeKind get_composite_kind() { return api::detail::get_value<CompositeKindParam>(); }
inline void set_composite_kind(CompositeKind value) { api::detail::set_value<CompositeKindParam>(value); }

inline const CompositeRGB get_composite_rgb() { return api::detail::get_value<CompositeRGBParam>(); }
inline const CompositeHSV get_composite_hsv() { return api::detail::get_value<CompositeHSVParam>(); }

inline TriggerChangeValue<CompositeRGB> change_composite_rgb()
{
    return api::detail::change_value<CompositeRGBParam>();
}
inline TriggerChangeValue<CompositeHSV> change_composite_hsv()
{
    return api::detail::change_value<CompositeHSVParam>();
}

inline bool get_composite_auto_weights() { return api::detail::get_value<CompositeAutoWeights>(); }
inline void set_composite_auto_weights(bool value) { api::detail::set_value<CompositeAutoWeights>(value); }

} // namespace holovibes::api
