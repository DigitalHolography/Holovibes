#ifndef XIQ_HH
# define XIQ_HH

# include "camera.hh"

# include <Windows.h>
# include <xiApi.h>

namespace camera
{
  class XiqCamera : Camera
  {
  public:
    XiqCamera()
      : Camera("Xiq")
      , device_(nullptr)
      , status_(XI_OK)
    {}

    ~XiqCamera()
    {}

    virtual bool init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;

  private:
    HANDLE device_;
    XI_RETURN status_;
  };
}

#endif /* !XIQ_HH */
