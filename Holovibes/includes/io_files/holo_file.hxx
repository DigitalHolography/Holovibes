/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
