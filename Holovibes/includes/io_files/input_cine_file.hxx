#pragma once

#include "input_cine_file.hh"

namespace holovibes::io_files
{
inline size_t InputCineFile::get_total_nb_frames() const { return CineFile::get_total_nb_frames(); }
} // namespace holovibes::io_files
