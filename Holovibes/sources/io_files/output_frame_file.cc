#include <assert.h>
#include "output_frame_file.hh"
#include "holovibes.hh"

namespace holovibes::io_files
{

OutputFrameFile::OutputFrameFile(const std::string& file_path)
        : FrameFile(file_path, FrameFile::OpeningMode::WRITE)
{
    // if (setvbuf(file_, NULL, _IOFBF, 512 * 512 * 2048) != 0)
    //     throw std::runtime_error("Incorrect type or size of output file's buffer");
}

double OutputFrameFile::compute_output_fps()
{
    double input_fps = static_cast<double>(GSH::instance().get_input_fps());
    double time_stride = static_cast<double>(GSH::instance().get_time_stride());

    assert(time_stride != 0);
    double output_fps = input_fps / time_stride;

    return output_fps;
}
} // namespace holovibes::io_files
