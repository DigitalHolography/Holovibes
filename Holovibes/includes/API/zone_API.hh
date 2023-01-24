#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline units::RectFd get_signal_zone() { return api::detail::get_value<SignalZone>(); };
inline units::RectFd get_noise_zone() { return api::detail::get_value<NoiseZone>(); };
inline units::RectFd get_composite_zone() { return api::detail::get_value<CompositeZone>(); };
inline units::RectFd get_zoomed_zone() { return api::detail::get_value<ZoomedZone>(); };
inline units::RectFd get_reticle_zone() { return api::detail::get_value<ReticleZone>(); };

inline void set_signal_zone(const units::RectFd& rect) { api::detail::set_value<SignalZone>(rect); };
inline void set_noise_zone(const units::RectFd& rect) { api::detail::set_value<NoiseZone>(rect); };
inline void set_composite_zone(const units::RectFd& rect) { api::detail::set_value<CompositeZone>(rect); };
inline void set_zoomed_zone(const units::RectFd& rect) { api::detail::set_value<ZoomedZone>(rect); };
inline void set_reticle_zone(const units::RectFd& rect) { api::detail::set_value<ReticleZone>(rect); };

inline TriggerChangeValue<units::RectFd> change_signal_zone() { return api::detail::change_value<SignalZone>(); };
inline TriggerChangeValue<units::RectFd> change_noise_zone() { return api::detail::change_value<NoiseZone>(); };
inline TriggerChangeValue<units::RectFd> change_composite_zone() { return api::detail::change_value<CompositeZone>(); };
inline TriggerChangeValue<units::RectFd> change_zoomed_zone() { return api::detail::change_value<ZoomedZone>(); };
inline TriggerChangeValue<units::RectFd> change_reticle_zone() { return api::detail::change_value<ReticleZone>(); };

} // namespace holovibes::api
