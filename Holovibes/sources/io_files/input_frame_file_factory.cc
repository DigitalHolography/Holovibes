#include "input_frame_file_factory.hh"
#include "input_holo_file.hh"
#include "input_cine_file.hh"
#include "spdlog/spdlog.h"

namespace holovibes::io_files
{
InputFrameFile* InputFrameFileFactory::open(const std::string& file_path)
{
    if (file_path.ends_with(".holo"))
        return new InputHoloFile(file_path);

    else if (file_path.ends_with(".cine"))
        return new InputCineFile(file_path);

    else
    {
        spdlog::critical("file path : {} not valid", file_path);
        throw FileException("Invalid file extension", false);
    }
}
} // namespace holovibes::io_files
