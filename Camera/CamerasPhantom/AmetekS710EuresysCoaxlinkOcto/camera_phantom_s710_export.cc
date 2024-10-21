#include "camera_phantom_s710.hh"

namespace camera
{

inline ICamera* new_camera_device() { return InitCam<CameraPhantom710>(); }

} // namespace camera