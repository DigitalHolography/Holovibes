#pragma once

#include "camera_phantom_interface.hh"

namespace camera
{
class EHoloGrabber980 : public EHoloGrabberInt
{
  public:
    EHoloGrabber980(Euresys::EGenTL& gentl,
                    unsigned int buffer_part_count,
                    std::string pixel_format,
                    unsigned int nb_grabbers);

    void setup(const CameraParamMap& params, Euresys::EGenTL& gentl) override;
};

class CameraPhantom980 : public CameraPhantomInt
{
  public:
    CameraPhantom980();
    void init_camera() override;
    void load_default_params() override;
};
} // namespace camera
