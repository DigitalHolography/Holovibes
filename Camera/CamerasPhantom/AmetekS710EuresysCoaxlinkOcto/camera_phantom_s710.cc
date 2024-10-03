#include "camera_phantom_s710.hh"

namespace camera
{
EHoloGrabber::EHoloGrabber(EGenTL& gentl, unsigned int buffer_part_count, std::string& pixel_format)
    : EHoloGrabberInt(gentl, buffer_part_count, pixel_format, 0) // TODO explain
{
    nb_grabbers_ = gentl.size();                   // TODO !!!!!
    if (available_grabbers_.size() < nb_grabbers_) // 2 or 4
    {                                              // TODO
        Logger::camera()->error("Not enough frame grabbers  connected to the camera, expected: {} but got: {}.",
                                nb_grabbers_,
                                available_grabbers_.size());
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

void EHoloGrabber::setup(const SetupParam& param, const std::string& fan_ctrl)
{
    // dynamic detection of the number of banks available.
    if (nb_grabbers == 0)
        nb_grabbers = grabbers_.length(); // TODO DES GUEUX

    if (nb_grabbers == 2)
    {
        available_grabbers_[0]->setString<RemoteModule>("Banks", "Banks_AB");
    }
    else if (nb_grabbers == 4)
    {
        available_grabbers_[0]->setString<RemoteModule>("Banks", "Banks_ABCD");
    } // else Error // TODO

    EHoloGrabberInt::setup(param);
    if (param.triggerSource == "SWTRIGGER")
        available_grabbers_[0]->setString<RemoteModule>("TimeStamp", "TSOff");
    available_grabbers_[0]->setString<RemoteModule>("FanCtrl", fan_ctrl);
}

} // namespace camera