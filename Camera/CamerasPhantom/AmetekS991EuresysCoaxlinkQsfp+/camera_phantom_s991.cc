#include "camera_phantom_s991.hh"

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
        Logger::camera()->error("Not enough frame grabbers  connected to the camera, expected: {} but got: {}.",
                                nb_grabbers_,
                                available_grabbers_.size());
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

void EHoloGrabber::setup(const SetupParam& param, unsigned int acquisition_frame_rate)
{
    available_grabbers_[0]->setString<RemoteModule>("Banks", "Banks_AB");
    EHoloGrabberInt::setup(param);

    if (triggerSource == "SWTRIGGER")
        available_grabbers_[0]->setInteger<RemoteModule>("AcquisitionFrameRate", acquisition_frame_rate);
}

CameraPhantom::CameraPhantom()
    : CameraPhantomInt("ametek_s991_euresys_coaxlink_qsfp+.ini", "s991")
    , : name_("Phantom S991")
{
}

void CameraPhantom::init_camera()
{
    EHoloGrabberInt::SetupParam param = {
        .full_height = full_height_,
        .width = width_,
        .nb_grabbers = nb_grabbers_,
        .stripe_height = 4,
        .stripe_arrangement = "Geometry_1X_1Y",
        .trigger_source = trigger_source_,
        .block_height = 0,
        .offsets = stripe_offsets_,
        .trigger_mode = trigger_mode_,
        .trigger_selector = trigger_selector_,
        .cycle_minimum_period = cycle_minimum_period_,
        .exposure_time = exposure_time_,
        .gain_selector = gain_selector_,
        .gain = gain_,
        .balance_white_marker = balance_white_marker_,
    };
    init_camera_(param);
}

ICamera* new_camera_device() = { return new CameraPhantom() };
} // namespace camera

} // namespace camera