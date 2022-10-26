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
    TriggerChangeValue<CompositeRGB> res = api::detail::change_value<CompositeRGBParam>();
    auto callback = res.callback_;
    res.callback_ = [callback]()
    {
        callback();
        GSH::instance().notify();
    };
    return TriggerChangeValue<CompositeRGB>{res};
}
inline TriggerChangeValue<CompositeHSV> change_composite_hsv()
{
    TriggerChangeValue<CompositeHSV> res = api::detail::change_value<CompositeHSVParam>();
    auto callback = res.callback_;
    res.callback_ = [callback]()
    {
        callback();
        GSH::instance().notify();
    };
    return TriggerChangeValue<CompositeHSV>{res};
}

inline bool get_composite_auto_weights() { return api::detail::get_value<CompositeAutoWeights>(); }
inline void set_composite_auto_weights(bool value) { api::detail::set_value<CompositeAutoWeights>(value); }

} // namespace holovibes::api
