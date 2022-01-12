#pragma once

#include "all_struct.hh"
#include "rect.hh"

namespace holovibes::units
{

struct Zones
{
    RectFd signal_zone;
    RectFd noise_zone;
    RectFd composite_zone;
    RectFd zoomed_zone;
    RectFd reticle_zone;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Zones)

} // namespace holovibes::units