#pragma once

#include "output_frame_file.hh"

namespace holovibes::io_files
{
inline OutputFrameFile::OutputFrameFile(const std::string& file_path)
    : FrameFile(file_path, FrameFile::OpeningMode::WRITE)
{
}
} // namespace holovibes::io_files
