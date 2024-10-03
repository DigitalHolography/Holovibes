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
        : CameraPhantomInt("ametek_s710_euresys_coaxlink_octo.ini", "s710")
        , _name("Phantom S710")
    {
    }

    void init_camera()
    {
        EHoloGrabberInt::SetupParam param = {
            .full_height = full_height_,
            .width = width_,
            .nb_grabbers = nb_grabbers_,
            .stripe_height = 8,
            .stripe_arrangement = "Geometry_1X_1YM",
            .trigger_source = trigger_source_,
            .block_height = 8,
            .offsets = stripe_offsets_,
            .trigger_mode = trigger_mode_,
            .trigger_selector = trigger_selector_,
            .cycle_minimum_period = cycle_minimum_period_,
            .exposure_time = exposure_time_,
            .gain_selector = gain_selector_,
            .gain = gain_,
            .balance_white_marker = balance_white_marker_,
            .flat_field_correction = flat_field_correction_,
        };
        init_camera_(param);
    }
}
} // namespace camera