#pragma once

#include "camera_phantom_interface.hh"

namespace camera
{
class EHoloGrabber : public EHoloGrabberInt
{
  public:
    EHoloGrabber(EGenTL& gentl, unsigned int buffer_part_count, std::string& pixel_format);

    void setup(const SetupParam& param, const std::string& fan_ctrl);
}
} // namespace camera