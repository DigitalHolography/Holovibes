#pragma once

#include "output_frame_file.hh"

namespace holovibes::io_files
{
inline OutputFrameFile::OutputFrameFile(const std::string& file_path)
    : FrameFile(file_path, FrameFile::OpeningMode::WRITE)
{
}

inline void OutputFrameFile::set_make_square_output(bool make_square_output)
{
    // if the output is anamorphic and should be a square output
    if (make_square_output && fd_.width != fd_.height)
        max_side_square_output_ = std::max(fd_.width, fd_.height);
}
} // namespace holovibes::io_files
