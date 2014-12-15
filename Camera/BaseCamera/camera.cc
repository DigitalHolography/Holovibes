#include <camera.hh>
#include "camera_pimpl.hh"

namespace camera
{
  Camera::Camera(const char* const ini_filepath)
    : pimpl_(new CameraPimpl(ini_filepath))
  {}

  Camera::~Camera()
  {
    delete pimpl_;
  }
  
  const FrameDescriptor& Camera::get_frame_descriptor() const
  {
    return pimpl_->desc_;
  }

  const char* Camera::get_name() const
  {
    return pimpl_->name_.c_str();
  }

  const char* Camera::get_ini_path() const
  {
    return pimpl_->get_ini_path().c_str();
  }
}