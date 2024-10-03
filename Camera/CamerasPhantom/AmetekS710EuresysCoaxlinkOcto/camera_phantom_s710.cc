#include "camera_phantom_s710.hh"
#include "camera_exception.hh"
namespace camera
{

EHoloGrabber::EHoloGrabber(Euresys::EGenTL& gentl,
                           unsigned int buffer_part_count,
                           std::string& pixel_format,
                           unsigned int nb_grabbers)
    : EHoloGrabberInt(gentl, buffer_part_count, pixel_format, nb_grabbers)
{
    size_t available_grabbers_count = available_grabbers.size();

    // nb_grabbers = 0 means autodetect between 2 or 4
    if (nb_grabbers_ == 0 && available_grabbers_count >= 2)
        nb_grabbers_ = (available_grabbers_count >= 4) ? 4 : 2;

    // S710 only supports 2 and 4 frame grabbers setup
    if (nb_grabbers_ != 2 && nb_grabbers_ != 4)
    {
        Logger::camera()->error("Incompatible number of frame grabbers requested for camera S710, please check the ",
                                "NbGrabbers parameter of the S710 ini file");
        throw CameraException(CameraException::CANT_SET_CONFIG)
    }

    // Not enough frame grabbers compared to requested number
    if (available_grabbers_count < nb_grabbers_)
    {
        // If possible recover to the 2 grabbers setup (with a warning)
        if (available_grabbers_count == 2)
        {
            Logger::camera()->warning(
                "Not enough frame grabbers connected to the camera, switched to 2 frame grabbers setup. Please "
                "check your setup and the NbGrabbers parameter in S710 ini config file");
            return;
        }

        Logger::camera()->error("Not enough frame grabbers connected to the camera. Please check NbGrabber "
                                "parameter in S710 ini config file");
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

void EHoloGrabber::setup(const SetupParam& param)
{
    if (nb_grabbers == 2)
        available_grabbers_[0]->setString<RemoteModule>("Banks", "Banks_AB");
    else if (nb_grabbers == 4)
        available_grabbers_[0]->setString<RemoteModule>("Banks", "Banks_ABCD");

    EHoloGrabberInt::setup(param);
    if (param.triggerSource == "SWTRIGGER")
        available_grabbers_[0]->setString<RemoteModule>("TimeStamp", "TSOff");
    available_grabbers_[0]->setString<RemoteModule>("FanCtrl", );
}

CameraPhantom::CameraPhantom()
    : CameraPhantomInt("ametek_s710_euresys_coaxlink_octo.ini", "s710")
    , : name_("Phantom S710")
{
}

void CameraPhantom::init_camera()
{
    EHoloGrabberInt::SetupParam param = {
        .full_height = full_height_,
        .width = width_,
        .nb_grabbers = nb_grabbers_,
        .stripe_height = 8,
        .stripe_arrangement = "Geometry_1X_2YM",
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
        .fan_ctrl = fan_ctrl_,
    };
    init_camera_(param);
}

ICamera* new_camera_device() = { return new CameraPhantom() };
} // namespace camera

} // namespace camera