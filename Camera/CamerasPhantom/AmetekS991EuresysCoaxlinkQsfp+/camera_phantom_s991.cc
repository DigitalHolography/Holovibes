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

ICamera* new_camera_device() = { return new CameraPhantom() };
} // namespace camera

} // namespace camera