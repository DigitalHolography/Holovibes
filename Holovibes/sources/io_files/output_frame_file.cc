#include <assert.h>
#include "output_frame_file.hh"
#include "holovibes.hh"

namespace holovibes::io_files
{
double OutputFrameFile::compute_output_fps()
{
    ComputeDescriptor& cd = Holovibes::instance().get_cd();
    double input_fps = static_cast<double>(cd.input_fps);
    double time_transformation_stride = static_cast<double>(GSH::instance().time_transformation_stride_query().value);

    assert(time_transformation_stride != 0);
    double output_fps = input_fps / time_transformation_stride;

    return output_fps;
}
} // namespace holovibes::io_files
