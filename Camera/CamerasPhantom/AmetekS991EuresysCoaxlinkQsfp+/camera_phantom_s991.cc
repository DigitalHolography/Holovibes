#include "camera_phantom_s991.hh"

#include <EGrabber.h>
#include <EGrabbers.h>

#include "camera_exception.hh"

namespace camera
{
EHoloGrabber::EHoloGrabber(Euresys::EGenTL& gentl,
                           unsigned int buffer_part_count,
                           std::string& pixel_format,
                           unsigned int nb_grabbers)
    : EHoloGrabberInt(gentl, buffer_part_count, pixel_format, NB_GRABBER)
{
    if (available_grabbers_.size() < nb_grabbers_)
    { // TODO tkt
        Logger::camera()->error("Not enough frame grabbers connected to the camera, expected: {} but got: {}.",
                                nb_grabbers_,
                                available_grabbers_.size());
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

void EHoloGrabber::setup(const SetupParam& param, Euresys::EGenTL& gentl)
{
    available_grabbers_[0]->setString<Euresys::RemoteModule>("Banks", "Banks_AB");
    EHoloGrabberInt::setup(param, gentl);

    if (param.trigger_source == "SWTRIGGER")
        available_grabbers_[0]->setInteger<Euresys::RemoteModule>("AcquisitionFrameRate", param.acquisition_frame_rate);
}

CameraPhantom::CameraPhantom()
    : CameraPhantomInt("ametek_s991_euresys_coaxlink_qsfp+.ini", "s991")
{
    name_ = "Phantom S991";

    grabber_ = std::make_unique<EHoloGrabber>(*gentl_, buffer_part_count_, pixel_format_, nb_grabbers_);
    init_camera();
}

void CameraPhantom::init_camera()
{
    EHoloGrabberInt::SetupParam param = {
        .full_height = full_height_,
        .width = width_,
        .nb_grabbers = nb_grabbers_,
        .pixel_format = pixel_format_,
        .stripe_height = 4,
        .stripe_arrangement = "Geometry_1X_1Y",
        .trigger_source = trigger_source_,
        .block_height = 0,
        .offsets = stripe_offsets_,
        .trigger_mode = std::nullopt,
        .trigger_selector = trigger_selector_,
        .cycle_minimum_period = cycle_minimum_period_,
        .exposure_time = exposure_time_,
        .gain_selector = gain_selector_,
        .gain = gain_,
        .balance_white_marker = balance_white_marker_,
        .acquisition_frame_rate = acquisition_frame_rate_,
    };
    init_camera_(param);
}

ICamera* new_camera_device() { return new CameraPhantom(); }
} // namespace camera