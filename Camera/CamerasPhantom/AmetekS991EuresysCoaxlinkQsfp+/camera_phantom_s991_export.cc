#include "camera_phantom_s991.hh"

namespace camera
{

inline ICamera* new_camera_device() { return initCam<CameraPhantom991>(); }

} // namespace camera