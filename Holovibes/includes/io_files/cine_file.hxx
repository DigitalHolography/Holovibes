/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
