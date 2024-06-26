#include <assert.h>
#include "output_frame_file.hh"
#include "holovibes.hh"

namespace holovibes::io_files
{

// OutputFrameFile::OutputFrameFile(const std::string& file_path)
//         : FrameFile(file_path, FrameFile::OpeningMode::WRITE)
// {
//     // if (setvbuf(file_, NULL, _IOFBF, 512 * 512 * 2048) != 0)
//     //     throw std::runtime_error("Incorrect type or size of output file's buffer");
// }

double OutputFrameFile::compute_output_fps()
{
    // TODO(julesguillou): Remove very bad, this class should not need to have access to holovibes singleton
    double input_fps = static_cast<double>(Holovibes::instance().template get_setting<settings::InputFPS>().value);
    double time_stride = static_cast<double>(Holovibes::instance().template get_setting<settings::TimeStride>().value);

    assert(time_stride != 0);
    double output_fps = input_fps / time_stride;

    return output_fps;
}
} // namespace holovibes::io_files
