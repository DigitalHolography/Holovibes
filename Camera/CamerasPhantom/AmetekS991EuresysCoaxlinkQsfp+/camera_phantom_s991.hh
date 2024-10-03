#pragma once

#include "camera_phantom_interface.hh"

namespace camera
{
#define NB_GRABBER 2
class EHoloGrabber : public EHoloGrabberInt
{
  public:
    EHoloGrabber(EGenTL& gentl, unsigned int buffer_part_count, std::string& pixel_format, unsigned int nb_grabbers);

    void setup(const SetupParam& param) override;
}

class CameraPhantom : public CameraPhantomInt
{
  public:
    CameraPhantom()
        : CameraPhantomInt("ametek_s991_euresys_coaxlink_qsfp+.ini", "s991")
    {
    }
}
} // namespace camera