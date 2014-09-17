#ifndef XIQ_HH
# define XIQ_HH

# include "camera.hh"

namespace camera
{
  class XiqCamera : Camera
  {
  public:
    XiqCamera()
      : Camera("Xiq")
    {}
    ~XiqCamera()
    {}

    bool init_camera() override;
    void start_acquisition() override;
    void stop_acquisition() override;
    void shutdown_camera() override;

  private:
  };
}

#endif /* !XIQ_HH */
