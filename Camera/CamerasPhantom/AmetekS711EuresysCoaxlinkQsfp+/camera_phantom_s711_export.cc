#include "camera_phantom_s711.hh"

namespace camera
{

inline ICamera* new_camera_device() { return initCam<CameraPhantom711>(); }

} // namespace camera