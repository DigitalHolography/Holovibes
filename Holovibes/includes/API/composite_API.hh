#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

// notify()

inline CompositeKindEnum get_composite_kind() { return api::detail::get_value<CompositeKind>(); }
inline void set_composite_kind(CompositeKindEnum value) { api::detail::set_value<CompositeKind>(value); }

inline const CompositeRGBStruct get_composite_rgb() { return api::detail::get_value<CompositeRGB>(); }
inline const CompositeHSVStruct get_composite_hsv() { return api::detail::get_value<CompositeHSV>(); }

inline TriggerChangeValue<CompositeRGBStruct> change_composite_rgb()
{
    return api::detail::change_value<CompositeRGB>();
}
inline TriggerChangeValue<CompositeHSVStruct> change_composite_hsv()
{
    return api::detail::change_value<CompositeHSV>();
}

inline bool get_composite_auto_weights() { return api::detail::get_value<CompositeAutoWeights>(); }
inline void set_composite_auto_weights(bool value) { api::detail::set_value<CompositeAutoWeights>(value); }

} // namespace holovibes::api