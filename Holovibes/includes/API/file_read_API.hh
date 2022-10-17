#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline uint get_file_buffer_size() { return api::detail::get_value<FileBufferSize>(); }
inline void set_file_buffer_size(uint value) { api::detail::set_value<FileBufferSize>(value); }

} // namespace holovibes::api
