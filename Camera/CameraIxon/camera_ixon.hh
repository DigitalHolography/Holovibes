#ifndef CAMERA_IXON_HH
# define CAMERA_IXON_HH

# include <camera.hh>

namespace camera
{
  //!< Camera Ixon Zyla.
  class CameraIxon : public Camera
  {
  public:
    CameraIxon();
    virtual ~CameraIxon();

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual void* get_frame() override;

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

  private:
    int vertical_shift_speed_;
    int horizontal_shift_speed_;

    int gain_mode_;

    float kinetic_time_;

    unsigned short r_x;
    unsigned short r_y;

    unsigned short *output_image_;

    long device_handle;

    unsigned short* image_;

    int trigger_mode_;

    int shutter_close_;
    int shutter_open_;

    int ttl_;

    int shutter_mode_;
    /* FIXME: What acquisiton means ? */
    int acquisiton_mode_;
    int read_mode_;
  };
}
#endif /* !CAMERA_IXON_HH */