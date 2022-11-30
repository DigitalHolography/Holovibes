#include <assert.h>
#include "output_frame_file.hh"
#include "holovibes.hh"
#include "API.hh"

namespace holovibes::io_files
{
double OutputFrameFile::compute_output_fps()
{
    double input_fps = static_cast<double>(api::detail::get_value<InputFps>());
    double time_stride = static_cast<double>(api::detail::get_value<TimeStride>());

    assert(time_stride != 0);
    double output_fps = input_fps / time_stride;

    return output_fps;
}
} // namespace holovibes::io_files
