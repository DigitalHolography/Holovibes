#pragma once

#include "output_mp4_file.hh"

namespace holovibes::io_files
{
inline size_t OutputMp4File::get_total_nb_frames() const { return img_nb_; }
} // namespace holovibes::io_files
