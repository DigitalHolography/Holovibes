#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

// notify()

inline CompositeKindEnum get_composite_kind() { return api::detail::get_value<CompositeKind_PARAM>(); }
inline void set_composite_kind(CompositeKindEnum value) { api::detail::set_value<CompositeKind_PARAM>(value); }

inline const CompositeRGBStruct get_composite_rgb() { return api::detail::get_value<CompositeRGB_PARAM>(); }
inline const CompositeHSVStruct get_composite_hsv() { return api::detail::get_value<CompositeHSV_PARAM>(); }

inline TriggerChangeValue<CompositeRGBStruct> change_composite_rgb()
{
    TriggerChangeValue<CompositeRGBStruct> res = api::detail::change_value<CompositeRGB_PARAM>();
    auto callback = res.callback_;
    res.callback_ = [callback]()
    {
        callback();
        GSH::instance().notify();
    };
    return TriggerChangeValue<CompositeRGBStruct>{res};
}
inline TriggerChangeValue<CompositeHSVStruct> change_composite_hsv()
{
    TriggerChangeValue<CompositeHSVStruct> res = api::detail::change_value<CompositeHSV_PARAM>();
    auto callback = res.callback_;
    res.callback_ = [callback]()
    {
        callback();
        GSH::instance().notify();
    };
    return TriggerChangeValue<CompositeHSVStruct>{res};
}

inline bool get_composite_auto_weights() { return api::detail::get_value<CompositeAutoWeights>(); }
inline void set_composite_auto_weights(bool value) { api::detail::set_value<CompositeAutoWeights>(value); }

} // namespace holovibes::api
