#ifndef XIQ_HH
# define XIQ_HH

# include "camera.hh"

# include <Windows.h>
# include <xiApi.h>

namespace camera
{
  class XiqCamera : public Camera
  {
  public:
    XiqCamera();

    ~XiqCamera()
    {}

    virtual bool init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual void* get_frame() override;

  private:
    HANDLE device_;
    XI_RETURN status_;
    XI_IMG frame_;
  };
}

#endif /* !XIQ_HH */
