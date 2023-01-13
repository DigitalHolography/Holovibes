#include "API.hh"

namespace holovibes
{

void compute_output_fd(const FrameDescriptor& import_fd, ComputeModeEnum mode, ImageTypeEnum type)
{
    FrameDescriptor output_fd = import_fd;
    if (mode == ComputeModeEnum::Hologram)
    {
        output_fd.depth = 2;
        if (type == ImageTypeEnum::Composite)
            output_fd.depth = 6;
    }
    api::detail::set_value<OutputFrameDescriptor>(output_fd);
}

} // namespace holovibes