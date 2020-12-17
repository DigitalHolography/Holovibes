/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "frame_file.hh"

namespace holovibes::io_files
{
inline camera::FrameDescriptor FrameFile::get_frame_descriptor() { return fd_; }

inline const camera::FrameDescriptor& FrameFile::get_frame_descriptor() const
{
    return fd_;
}
} // namespace holovibes::io_files
