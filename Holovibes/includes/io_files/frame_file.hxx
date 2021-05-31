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
