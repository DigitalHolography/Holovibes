#pragma once

#include "cine_file.hh"

namespace holovibes::io_files
{
inline CineFile::~CineFile() {}

inline size_t CineFile::get_total_nb_frames() const
{
    return cine_file_header_.image_count;
}
} // namespace holovibes::io_files
