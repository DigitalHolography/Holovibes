#pragma once

#include "camera_phantom_interface.hh"

namespace camera
{

class EHoloGrabber : public EHoloGrabberInt
{
  public:
    EHoloGrabber(Euresys::EGenTL& gentl,
                 unsigned int buffer_part_count,
                 std::string& pixel_format,
                 unsigned int& nb_grabbers);

    void setup(const SetupParam& param, Euresys::EGenTL& gentl) override;
};

class CameraPhantom : public CameraPhantomInt
{
  public:
    CameraPhantom();
    virtual void init_camera() override;
};
} // namespace camera