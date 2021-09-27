#pragma once

#include "input_frame_file.hh"

namespace holovibes::io_files
{
inline InputFrameFile::InputFrameFile(const std::string& file_path)
    : FrameFile(file_path, FrameFile::OpeningMode::READ)
{
}
} // namespace holovibes::io_files
