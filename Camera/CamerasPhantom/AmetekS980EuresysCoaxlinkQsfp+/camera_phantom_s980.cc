#include "camera_phantom_s980.hh"

#include "camera_exception.hh"

namespace camera
{
EHoloGrabber980::EHoloGrabber980(Euresys::EGenTL& gentl,
                                 unsigned int buffer_part_count,
                                 std::string pixel_format,
                                 unsigned int nb_grabbers)
    : EHoloGrabberInt(gentl, buffer_part_count, pixel_format, nb_grabbers)
{
    if (nb_grabbers_ == 0)
        nb_grabbers_ = static_cast<unsigned int>(available_grabbers_.size() >= 2 ? 2 : 1);

    if (nb_grabbers_ != 1 && nb_grabbers_ != 2)
    {
        Logger::camera()->error("Incompatible number of frame grabbers for Phantom S980; expected 1 or 2");
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
    if (available_grabbers_.size() < nb_grabbers_)
    {
        Logger::camera()->error("Not enough frame grabbers connected for Phantom S980; expected {}, got {}",
                                nb_grabbers_,
                                available_grabbers_.size());
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

void EHoloGrabber980::setup(const CameraParamMap& params, Euresys::EGenTL& gentl)
{
    available_grabbers_[0]->setString<Euresys::RemoteModule>("Banks", nb_grabbers_ > 1 ? "Banks_AB" : "Banks_A");
    EHoloGrabberInt::setup(params, gentl);
}

CameraPhantom980::CameraPhantom980()
    : CameraPhantomInt("ametek_s980_euresys_coaxlink_qsfp+.ini", "s980")
{
    name_ = "Phantom S980";
}

void CameraPhantom980::load_default_params()
{
    CameraPhantomInt::load_default_params();
    params_.set<unsigned int>("StripeHeight", 4, false);
    params_.set<unsigned int>("BlockHeight", 0, false);
    params_.set<std::string>("StripeArrangement", "Geometry_1X_1Y", false);
    params_.set<std::string>("TriggerSelector", "");
}

void CameraPhantom980::init_camera()
{
    load_default_params();
    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    auto nb_grabbers = params_.at<unsigned int>("NbGrabbers");
    grabber_ = std::make_unique<EHoloGrabber980>(*gentl_,
                                                 params_.at<unsigned int>("BufferPartCount"),
                                                 params_.at<std::string>("PixelFormat"),
                                                 nb_grabbers);
    params_.set<unsigned int>("NbGrabbers", nb_grabbers);
    CameraPhantomInt::init_camera();
}
} // namespace camera
