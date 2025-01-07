#pragma once

#include "camera_phantom_interface.hh"

namespace camera
{

class EHoloGrabber711 : public EHoloGrabberInt
{
  public:
    EHoloGrabber711(Euresys::EGenTL& gentl,
                    unsigned int buffer_part_count,
                    std::string pixel_format,
                    unsigned int& nb_grabbers);

    void setup(const CameraParamMap& params, Euresys::EGenTL& gentl) override;
};
class CameraPhantom711 : public CameraPhantomInt
{
  public:
    CameraPhantom711();
    void init_camera() override;
    void load_default_params() override;
};
} // namespace camera