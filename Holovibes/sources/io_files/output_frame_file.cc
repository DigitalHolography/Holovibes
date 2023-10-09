#include <assert.h>
#include "output_frame_file.hh"
#include "holovibes.hh"

namespace holovibes::io_files
{
double OutputFrameFile::compute_output_fps()
{
    // TODO(julesguillou): Remove very bad, this class should not need to have access to holovibes singleton
    double input_fps = static_cast<double>(Holovibes::instance().template get_setting<settings::InputFPS>().value);
    double time_stride = static_cast<double>(GSH::instance().get_time_stride());

    assert(time_stride != 0);
    double output_fps = input_fps / time_stride;

    return output_fps;
}
} // namespace holovibes::io_files
