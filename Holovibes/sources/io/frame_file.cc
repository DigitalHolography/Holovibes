#include "frame_file.hh"
#include "file_exception.hh"

namespace holovibes::io_files
{
FrameFile::FrameFile(const std::string& file_path, FrameFile::OpeningMode mode)
    : file_path_(file_path)
{
    if (mode == FrameFile::OpeningMode::READ)
        file_ = fopen(file_path_.c_str(), "rb");

    else
        file_ = fopen(file_path_.c_str(), "wb");

    // if an error occurred
    if (file_ == nullptr)
        throw FileException("Unable to open file " + file_path_);
}

FrameFile::~FrameFile() { std::fclose(file_); }
} // namespace holovibes::io_files
