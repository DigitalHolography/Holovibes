/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "input_frame_file_factory.hh"
#include "input_holo_file.hh"
#include "input_cine_file.hh"

namespace holovibes::io_files
{
InputFrameFile* InputFrameFileFactory::open(const std::string& file_path)
{
    if (file_path.ends_with(".holo"))
        return new InputHoloFile(file_path);

    else if (file_path.ends_with(".cine"))
        return new InputCineFile(file_path);

    else
        throw FileException("Invalid file extension", false);
}
} // namespace holovibes::io_files
