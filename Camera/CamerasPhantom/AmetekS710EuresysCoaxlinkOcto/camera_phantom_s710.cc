#include "camera_phantom_s710.hh"
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

    // nb_grabbers = 0 means autodetect between 2 or 4
    if (nb_grabbers == 0 && available_grabbers_count >= 2)
        nb_grabbers = (available_grabbers_count >= 4) ? 4 : 2;
    nb_grabbers_ = nb_grabbers;

    // S710 only supports 2 and 4 frame grabbers setup
    if (nb_grabbers != 2 && nb_grabbers != 4)
    {
        Logger::camera()->error("Incompatible number of frame grabbers requested for camera S710, please check the "
                                "NbGrabbers parameter of the S710 ini file");
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }

    // Not enough frame grabbers compared to requested number
    if (available_grabbers_count < nb_grabbers)
    {
        // If possible recover to the 2 grabbers setup (with a warning)
        if (available_grabbers_count == 2)
        {
            Logger::camera()->warn(
                "Not enough frame grabbers connected to the camera, switched to 2 frame grabbers setup. Please "
                "check your setup and the NbGrabbers parameter in S710 ini config file");
            return;
        }

        Logger::camera()->error("Not enough frame grabbers connected to the camera. Please check NbGrabber "
                                "parameter in S710 ini config file");
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

void EHoloGrabber::setup(const CameraParamMap& params, Euresys::EGenTL& gentl)
{
    if (nb_grabbers_ == 2)
        available_grabbers_[0]->setString<Euresys::RemoteModule>("Banks", "Banks_AB");
    else if (nb_grabbers_ == 4)
        available_grabbers_[0]->setString<Euresys::RemoteModule>("Banks", "Banks_ABCD");

    EHoloGrabberInt::setup(params, gentl);

    if (params.at<std::string>("TriggerSource") == "SWTRIGGER")
        available_grabbers_[0]->setString<Euresys::RemoteModule>("TimeStamp", "TSOff");

    available_grabbers_[0]->setString<Euresys::RemoteModule>("FanCtrl", params.at<std::string>("FanCtrl"));
    available_grabbers_[0]->setString<Euresys::RemoteModule>("FlatFieldCorrection",
                                                             params.at<std::string>("FlatFieldCorrection"));
}

CameraPhantom::CameraPhantom()
    : CameraPhantomInt("ametek_s710_euresys_coaxlink_octo.ini", "s710")
{
    name_ = "Phantom S710";
}

void CameraPhantom::load_default_params()
{
    CameraPhantomInt::load_default_params();
    params_.set<std::string>("FanCtrl", "");
    params_.set<unsigned int>("StripeHeight", 8, false);
    params_.set<unsigned int>("BlockHeight", 8, false);
    params_.set<std::string>("StripeArrangement", "Geometry_1X_2YM", false);
    params_.set<std::string>("TriggerMode", "");
    params_.set<std::string>("FlatFieldCorrection", "");
    params_.set<std::string>("TimeStamp", "TSOff");
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