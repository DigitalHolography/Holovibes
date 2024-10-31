#pragma once

#include "camera_phantom_interface.hh"

namespace camera
{

class EHoloGrabber710 : public EHoloGrabberInt
{
  public:
    EHoloGrabber710(Euresys::EGenTL& gentl,
                    unsigned int buffer_part_count,
                    std::string pixel_format,
                    unsigned int& nb_grabbers);

    void setup(const CameraParamMap& param, Euresys::EGenTL& gentl) override;
};

class CameraPhantom710 : public CameraPhantomInt
{
  public:
    CameraPhantom710();
    virtual void init_camera() override;

  protected:
    virtual void load_default_params() override;
};
} // namespace camera