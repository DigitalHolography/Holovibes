#include "camera_phantom_interface.hh"

namespace camera
{
template <class Cam>
Cam* initCam<Cam>()
{
    auto* res = new Cam();
    res->init_camera();
    return res;
}

} // namespace camera