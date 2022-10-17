#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline uint get_start_frame() { return api::detail::get_value<StartFrame>(); }
inline void set_start_frame(uint value) { api::detail::set_value<StartFrame>(value); }

inline uint get_end_frame() { return api::detail::get_value<EndFrame>(); }
inline void set_end_frame(uint value) { api::detail::set_value<EndFrame>(value); }

bool import_start(
    std::string& file_path, unsigned int fps, size_t first_frame, bool load_file_in_gpu, size_t last_frame);

void import_stop();
std::optional<io_files::InputFrameFile*> import_file(const std::string& filename);

} // namespace holovibes::api
