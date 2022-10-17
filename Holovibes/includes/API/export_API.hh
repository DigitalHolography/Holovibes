#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline void get_frame_record_enabled(bool value) { api::detail::get_value<FrameRecordEnable>(); }
inline void set_frame_record_enabled(bool value) { api::detail::set_value<FrameRecordEnable>(value); }

} // namespace holovibes::api
