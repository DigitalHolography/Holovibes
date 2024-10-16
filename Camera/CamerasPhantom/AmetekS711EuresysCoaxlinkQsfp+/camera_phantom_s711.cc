#include "camera_phantom_s711.hh"
#include "camera_exception.hh"

namespace camera
{
EHoloGrabber::EHoloGrabber(Euresys::EGenTL& gentl,
                           unsigned int buffer_part_count,
                           std::string pixel_format,
                           unsigned int& nb_grabbers)
    : EHoloGrabberInt(gentl, buffer_part_count, pixel_format, nb_grabbers)
{
    size_t available_grabbers_count = available_grabbers_.size();

    // nb_grabbers = 0 means autodetect between 1 or 2
    if (nb_grabbers == 0 && available_grabbers_count != 0)
        nb_grabbers = (available_grabbers_count >= 2) ? 2 : 1;
    nb_grabbers_ = nb_grabbers;

    // S711 only supports 1 and 2 frame grabbers setup
    if (nb_grabbers != 1 && nb_grabbers != 2)
    {
        Logger::camera()->error("Incompatible number of frame grabbers requested for camera S711, please check the "
                                "NbGrabbers parameter of the S711 ini file");
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }

    // Not enough frame grabbers compared to requested number
    if (available_grabbers_count < nb_grabbers)
    {
        // If possible recover to the 1 grabbers setup (with a warning)
        if (available_grabbers_count == 1)
        {
            Logger::camera()->warn(
                "Not enough frame grabbers connected to the camera, switched to 1 frame grabbers setup. Please "
                "check your setup and the NbGrabbers parameter in S711 ini config file");
            return;
        }

        Logger::camera()->error("Not enough frame grabbers connected to the camera. Please check NbGrabber "
                                "parameter in S711 ini config file");
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

void EHoloGrabber::setup(const CameraParamMap& params, Euresys::EGenTL& gentl)
{
    if (nb_grabbers_ > 1)
        available_grabbers_[0]->setString<Euresys::RemoteModule>("Banks", "Banks_AB");
    else
        available_grabbers_[0]->setString<Euresys::RemoteModule>("Banks", "Banks_A");

    EHoloGrabberInt::setup(params, gentl);

    available_grabbers_[0]->setString<Euresys::RemoteModule>("FlatFieldCorrection",
                                                             params.at<std::string>("FlatFieldCorrection"));
}

CameraPhantom::CameraPhantom()
    : CameraPhantomInt("ametek_s711_euresys_coaxlink_qsfp+.ini", "s711")
{
    name_ = "Phantom S711";
}

void CameraPhantom::load_default_params()
{
    CameraPhantomInt::load_default_params();
    params_.set<unsigned int>("StripeHeight", 8, false);
    params_.set<unsigned int>("BlockHeight", 8, false);
    params_.set<std::string>("StripeArrangement", "Geometry_1X_2YM", false);
    params_.set<std::string>("TriggerSelector", "");
    params_.set<std::string>("TriggerMode", "");
    params_.set<std::string>("FlatFieldCorrection", "");
}

void CameraPhantom::init_camera()
{
    load_default_params();

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    unsigned int nb_grabbers = params_.at<unsigned int>("NbGrabbers");
    grabber_ = std::make_unique<EHoloGrabber>(*gentl_,
                                              params_.at<unsigned int>("BufferPartCount"),
                                              params_.at<std::string>("PixelFormat"),
                                              nb_grabbers);

    // nb_grabbers may have been updated by EHoloGrabber constructor
    params_.set<unsigned int>("NbGrabbers", nb_grabbers);

    CameraPhantomInt::init_camera();
}

ICamera* new_camera_device()
{
    auto* res = new CameraPhantom();
    res->init_camera();
    return res;
}

} // namespace camera