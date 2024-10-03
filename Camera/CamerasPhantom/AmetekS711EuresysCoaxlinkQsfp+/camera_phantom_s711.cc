#include "camera_phantom_s711.hh"

namespace camera
{
EHoloGrabber::EHoloGrabber(EGenTL& gentl,
                           unsigned int buffer_part_count,
                           std::string& pixel_format,
                           unsigned int nb_grabbers)
    : EHoloGrabberInt(gentl, buffer_part_count, pixel_format, nb_grabbers)
{
    if (available_grabbers_.size() < nb_grabbers_)
    { // TODO
        Logger::camera()->error("Not enough frame grabbers  connected to the camera, expected: {} but got: {}.",
                                nb_grabbers_,
                                available_grabbers_.size());
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

void EHoloGrabber::setup(const SetupParam& param)
{
    if (available_grabbers_.size() > 1)
        available_grabbers_[0]->setString<RemoteModule>("Banks", "Banks_AB");
    else
        available_grabbers_[0]->setString<RemoteModule>("Banks", "Banks_A");
    EHoloGrabberInt::setup(param);
    available_grabbers_[0]->setString<RemoteModule>("FlatFieldCorrection", flat_field_correction);
}

} // namespace camera