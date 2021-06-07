#pragma once

#include "output_holo_file.hh"

namespace holovibes::io_files
{
inline size_t OutputHoloFile::get_total_nb_frames() const
{
    return HoloFile::get_total_nb_frames();
}
} // namespace holovibes::io_files
