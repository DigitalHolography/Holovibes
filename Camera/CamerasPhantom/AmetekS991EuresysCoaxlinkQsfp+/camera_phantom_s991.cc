#include "camera_phantom_s991.hh"

#include <EGrabber.h>
#include <EGrabbers.h>

#include "camera_exception.hh"

namespace camera
{
EHoloGrabber991::EHoloGrabber991(Euresys::EGenTL& gentl,
                                 unsigned int buffer_part_count,
                                 std::string pixel_format,
                                 unsigned int nb_grabbers)
    : EHoloGrabberInt(gentl, buffer_part_count, pixel_format, nb_grabbers_)
{
    if (available_grabbers_.size() < nb_grabbers_)
    {
        Logger::camera()->error(
            "Not enough frame grabbers connected to the camera, expected: {} (from the ini config file) but got: {}.",
            nb_grabbers_,
            available_grabbers_.size());
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

void EHoloGrabber991::setup(const CameraParamMap& params, Euresys::EGenTL& gentl)
{
    available_grabbers_[0]->setString<Euresys::RemoteModule>("Banks", "Banks_AB");
    EHoloGrabberInt::setup(params, gentl);

    if (params.at<std::string>("TriggerSource") == "SWTRIGGER")
        available_grabbers_[0]->setString<Euresys::RemoteModule>("AcquisitionFrameRate",
                                                                 params.at<std::string>("AcquisitionFrameRate"));
}

CameraPhantom991::CameraPhantom991()
    : CameraPhantomInt("ametek_s991_euresys_coaxlink_qsfp+.ini", "s991")
{
    name_ = "Phantom S991";
}

void CameraPhantom991::load_default_params()
{
    CameraPhantomInt::load_default_params();
    params_.set<unsigned int>("StripeHeight", 4, false);
    params_.set<unsigned int>("BlockHeight", 0, false);
    params_.set<std::string>("StripeArrangement", "Geometry_1X_1Y", false);
    params_.set<std::string>("AcquisitionFrameRate", "");
    params_.set<std::string>("TriggerSelector", "");
}

void CameraPhantom991::init_camera()
{
    load_default_params();

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    grabber_ = std::make_unique<EHoloGrabber991>(*gentl_,
                                                 params_.at<unsigned int>("BufferPartCount"),
                                                 params_.at<std::string>("PixelFormat"),
                                                 params_.at<unsigned int>("NbGrabbers"));

    CameraPhantomInt::init_camera();
}

} // namespace camera