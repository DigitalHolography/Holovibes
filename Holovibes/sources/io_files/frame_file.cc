/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
