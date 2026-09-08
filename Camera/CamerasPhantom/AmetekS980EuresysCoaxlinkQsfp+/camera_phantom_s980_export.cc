#include "camera_phantom_s980.hh"

namespace camera
{
inline ICamera* new_camera_device() { return InitCam<CameraPhantom980>(); }
} // namespace camera
