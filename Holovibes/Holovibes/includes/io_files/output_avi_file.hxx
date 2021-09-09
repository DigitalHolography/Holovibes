#pragma once

#include "output_avi_file.hh"

namespace holovibes::io_files
{
inline size_t OutputAviFile::get_total_nb_frames() const { return img_nb_; }
} // namespace holovibes::io_files
