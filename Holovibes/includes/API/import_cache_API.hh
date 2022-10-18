#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline uint get_start_frame() { return api::detail::get_value<StartFrame>(); }
inline void set_start_frame(uint value) { api::detail::set_value<StartFrame>(value); }

inline uint get_end_frame() { return api::detail::get_value<EndFrame>(); }
inline void set_end_frame(uint value) { api::detail::set_value<EndFrame>(value); }

} // namespace holovibes::api
