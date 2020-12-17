/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "input_frame_file.hh"

namespace holovibes::io_files
{
inline InputFrameFile::InputFrameFile(const std::string& file_path)
    : FrameFile(file_path, FrameFile::OpeningMode::READ)
{
}
} // namespace holovibes::io_files
