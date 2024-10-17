#include <iostream>

#include "camera_phantom_s711.hh"
#include "camera_phantom_s710.hh"
#include "camera_phantom_s991.hh"
#include "camera_exception.hh"

namespace camera
{

namespace
{
static auto camera_init_map = std::map<std::string, std::function<ICamera*()>>{
    {"Phantom S710", initCam<CameraPhantom710>},
    {"Phantom S711", initCam<CameraPhantom711>},
    {"Phantom S991", initCam<CameraPhantom991>},
};
} // namespace

ICamera* new_camera_device()
{
    std::string model_name;
    {
        Euresys::EGenTL gentl;
        Euresys::EGrabber<> grabber(gentl, 0, 0);
        model_name = grabber.getString<Euresys::DeviceModule>("DeviceModelName");
    }
    if (camera_init_map.contains(model_name))
        return camera_init_map[model_name]();

    Logger::camera()->error(
        model_name + " camera cannot be detected by this auto detection try with an other way to connect your camera.");
    throw CameraException(CameraException::CANT_SET_CONFIG);
}
} // namespace camera