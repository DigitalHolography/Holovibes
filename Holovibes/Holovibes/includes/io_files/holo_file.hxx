#pragma once

#include "holo_file.hh"

namespace holovibes::io_files
{
inline HoloFile::~HoloFile() {}

inline size_t HoloFile::get_total_nb_frames() const
{
    return holo_file_header_.img_nb;
}
} // namespace holovibes::io_files
