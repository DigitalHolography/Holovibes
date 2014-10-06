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
    void load_param();
  private:
    HANDLE device_;
    XI_IMG frame_;

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
  };
}

#endif /* !XIQ_HH */
